import os
import time
import uuid
import gc
import threading
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

import torch
from transformers import AutoModel
from huggingface_hub import snapshot_download
from unsloth import FastVisionModel


# -----------------------------
# Config (Level 1 lazy loading)
# -----------------------------
os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"

MODEL_ID = os.getenv("DS_OCR2_MODEL_ID", "unsloth/DeepSeek-OCR-2")
SNAPSHOT_DIR = os.path.expanduser(
    os.getenv("DS_OCR2_SNAPSHOT_DIR", "~/models/deepseek_ocr2_unsloth")
)
OUTPUT_ROOT = os.path.expanduser(
    os.getenv("DS_OCR2_OUTPUT_ROOT", "/tmp/deepseek_ocr2_out")
)

# Level 1 behavior: keep process alive, unload model after idle
IDLE_UNLOAD_SECONDS = int(
    os.getenv("DS_OCR2_IDLE_UNLOAD_SECONDS", "900")
)  # 15 min default
WATCHDOG_POLL_SECONDS = int(os.getenv("DS_OCR2_WATCHDOG_POLL_SECONDS", "10"))

# Inference defaults (DeepSeek examples commonly use base_size=1024, image_size=768)
DEFAULT_BASE_SIZE = int(os.getenv("DS_OCR2_BASE_SIZE", "1024"))
DEFAULT_IMAGE_SIZE = int(os.getenv("DS_OCR2_IMAGE_SIZE", "768"))
DEFAULT_CROP_MODE = os.getenv("DS_OCR2_CROP_MODE", "1").lower() in ("1", "true", "yes")

# Optional (VRAM saving) – Unsloth supports 4bit loading flag in from_pretrained
LOAD_IN_4BIT = os.getenv("DS_OCR2_LOAD_IN_4BIT", "0").lower() in ("1", "true", "yes")

# Server
HOST = os.getenv("DS_OCR2_HOST", "0.0.0.0")
PORT = int(os.getenv("DS_OCR2_PORT", "8012"))


# -----------------------------
# Global model state
# -----------------------------
_state_lock = threading.Lock()
_infer_lock = threading.Lock()

_model = None
_tokenizer = None
_loaded_at: Optional[float] = None
_last_used: Optional[float] = None
_total_requests = 0


def _snapshot_ready() -> bool:
    return os.path.isdir(SNAPSHOT_DIR) and any(
        os.path.exists(os.path.join(SNAPSHOT_DIR, fn))
        for fn in ("config.json", "model.safetensors", "model.safetensors.index.json")
    )


def _ensure_snapshot():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    if not _snapshot_ready():
        snapshot_download(MODEL_ID, local_dir=SNAPSHOT_DIR)


def _load_model():
    global _model, _tokenizer, _loaded_at, _last_used

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. DeepSeek-OCR-2 infer path here expects GPU (uses .cuda())."
        )

    _ensure_snapshot()

    model, tokenizer = FastVisionModel.from_pretrained(
        SNAPSHOT_DIR,
        load_in_4bit=LOAD_IN_4BIT,
        auto_model=AutoModel,
        trust_remote_code=True,
        unsloth_force_compile=True,
        use_gradient_checkpointing="unsloth",
    )

    model.eval()
    model = model.cuda()

    _model = model
    _tokenizer = tokenizer
    _loaded_at = time.time()
    _last_used = time.time()


def _unload_model():
    global _model, _tokenizer, _loaded_at

    m = None
    t = None

    with _state_lock:
        if _model is None:
            return
        m = _model
        t = _tokenizer
        _model = None
        _tokenizer = None
        _loaded_at = None

    # best-effort VRAM release
    del m, t
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _ensure_loaded():
    with _state_lock:
        already = _model is not None
    if already:
        return

    # single-flight load
    with _infer_lock:
        with _state_lock:
            if _model is not None:
                return
        _load_model()


def _touch_used():
    global _last_used, _total_requests
    with _state_lock:
        _last_used = time.time()
        _total_requests += 1


def _watchdog():
    while True:
        time.sleep(WATCHDOG_POLL_SECONDS)
        with _state_lock:
            loaded = _model is not None
            last = _last_used
        if loaded and last and (time.time() - last) >= IDLE_UNLOAD_SECONDS:
            _unload_model()


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="DeepSeek-OCR-2 (Unsloth) - Lazy Level 1")


@app.on_event("startup")
def startup():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    th = threading.Thread(target=_watchdog, daemon=True)
    th.start()


@app.get("/health")
def health():
    with _state_lock:
        loaded = _model is not None
        loaded_at = _loaded_at
        last_used = _last_used
        total = _total_requests
    return {
        "ok": True,
        "loaded": loaded,
        "loaded_at": loaded_at,
        "last_used": last_used,
        "idle_unload_seconds": IDLE_UNLOAD_SECONDS,
        "total_requests": total,
        "load_in_4bit": LOAD_IN_4BIT,
        "snapshot_dir": SNAPSHOT_DIR,
    }


@app.post("/admin/unload")
def admin_unload():
    _unload_model()
    return {"ok": True, "unloaded": True}


@app.post("/v1/ocr")
async def ocr(
    file: UploadFile = File(...),
    mode: str = Form("markdown"),  # "markdown" or "free"
    prompt: Optional[str] = Form(None),
    base_size: int = Form(DEFAULT_BASE_SIZE),
    image_size: int = Form(DEFAULT_IMAGE_SIZE),
    crop_mode: bool = Form(DEFAULT_CROP_MODE),
    keep_files: bool = Form(False),
):
    # Prompts per DeepSeek examples
    if prompt is None:
        if mode.lower() == "free":
            prompt = "<image>\nFree OCR. "
        else:
            prompt = "<image>\n<|grounding|>Convert the document to markdown. "

    req_id = uuid.uuid4().hex
    req_dir = os.path.join(OUTPUT_ROOT, req_id)
    os.makedirs(req_dir, exist_ok=True)

    img_path = os.path.join(req_dir, file.filename or "image")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty upload")

    with open(img_path, "wb") as f:
        f.write(content)

    try:
        _ensure_loaded()
        _touch_used()

        # Serialize heavy GPU inference
        with _infer_lock:
            with _state_lock:
                model = _model
                tok = _tokenizer
            if model is None or tok is None:
                raise RuntimeError("Model not loaded (unexpected)")

            # Use eval_mode=True so infer returns the text (see model code path)
            text = model.infer(
                tok,
                prompt=prompt,
                image_file=img_path,
                output_path=req_dir,
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode,
                save_results=False,
                test_compress=False,
                eval_mode=True,
            )

        # Save result ourselves (eval_mode returns before model's internal save_results path)
        out_file = os.path.join(req_dir, "result.txt")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(text)

        resp = {
            "id": req_id,
            "text": text,
            "files": {"image": img_path, "result_txt": out_file}
            if keep_files
            else {"result_txt": out_file},
            "loaded_in_4bit": LOAD_IN_4BIT,
        }

        if not keep_files:
            # keep only result.txt by default
            for name in os.listdir(req_dir):
                p = os.path.join(req_dir, name)
                if os.path.isdir(p):
                    # remove generated folders like images/
                    try:
                        import shutil

                        shutil.rmtree(p, ignore_errors=True)
                    except Exception:
                        pass
                elif name not in ("result.txt",):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

        return JSONResponse(resp)

    except Exception as e:
        # Best-effort cleanup
        if not keep_files:
            try:
                import shutil

                shutil.rmtree(req_dir, ignore_errors=True)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
