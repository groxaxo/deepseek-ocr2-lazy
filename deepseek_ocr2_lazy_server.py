import os
import time
import uuid
import gc
import threading
import argparse
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

# -----------------------------
# Config
# -----------------------------
os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"

MODEL_ID = os.getenv("DS_OCR2_MODEL_ID", "unsloth/DeepSeek-OCR-2")
SNAPSHOT_DIR = os.path.expanduser(
    os.getenv("DS_OCR2_SNAPSHOT_DIR", "~/models/deepseek_ocr2_unsloth")
)
OUTPUT_ROOT = os.path.expanduser(
    os.getenv("DS_OCR2_OUTPUT_ROOT", "/tmp/deepseek_ocr2_out")
)

IDLE_UNLOAD_SECONDS = int(
    os.getenv("DS_OCR2_IDLE_UNLOAD_SECONDS", "900")
)  # 15 min default
WATCHDOG_POLL_SECONDS = int(os.getenv("DS_OCR2_WATCHDOG_POLL_SECONDS", "10"))
LOAD_IN_4BIT = os.getenv("DS_OCR2_LOAD_IN_4BIT", "0").lower() in ("1", "true", "yes")
HOST = os.getenv("DS_OCR2_HOST", "0.0.0.0")
PORT = int(os.getenv("DS_OCR2_PORT", "8012"))

# Check for Mock Mode
MOCK_MODE = os.getenv("DS_OCR2_MOCK", "0").lower() in ("1", "true", "yes")

# Imports that might fail on non-GPU/Unsloth setups
if not MOCK_MODE:
    import torch
    from transformers import AutoModel
    from huggingface_hub import snapshot_download
    from unsloth import FastVisionModel
else:
    print("!!! RUNNING IN MOCK MODE (No GPU required) !!!")
    torch = None
    snapshot_download = None
    FastVisionModel = None

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
    if MOCK_MODE:
        return True
    return os.path.isdir(SNAPSHOT_DIR) and any(
        os.path.exists(os.path.join(SNAPSHOT_DIR, fn))
        for fn in ("config.json", "model.safetensors", "model.safetensors.index.json")
    )


def _ensure_snapshot():
    if MOCK_MODE:
        return
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    if not _snapshot_ready():
        snapshot_download(MODEL_ID, local_dir=SNAPSHOT_DIR)


class MockModel:
    def eval(self):
        pass

    def cuda(self):
        return self

    def infer(self, *args, **kwargs):
        time.sleep(2)  # Simulate work
        return "## Mock OCR Result\n\nThis is a simulated response because the server is running in MOCK_MODE.\n\n- Item 1: Detected\n- Item 2: Text"


def _load_model():
    global _model, _tokenizer, _loaded_at, _last_used

    if not MOCK_MODE:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA not available. DeepSeek-OCR-2 infer path here expects GPU."
            )

        _ensure_snapshot()

        print("Loading real model...")
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
    else:
        print("Loading MOCK model...")
        time.sleep(1)  # Simulate load time
        model = MockModel()
        tokenizer = object()

    # Update globals safely
    with _state_lock:
        _model = model
        _tokenizer = tokenizer
        _loaded_at = time.time()
        _last_used = time.time()
    print("Model loaded.")


def _unload_model():
    global _model, _tokenizer, _loaded_at

    m = None
    t = None

    with _state_lock:
        if _model is None:
            return
        print("Unloading model due to idle...")
        m = _model
        t = _tokenizer
        _model = None
        _tokenizer = None
        _loaded_at = None

    # best-effort VRAM release
    del m, t
    gc.collect()
    if not MOCK_MODE and torch and torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    print("Model unloaded.")


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

        if loaded and last:
            idle_time = time.time() - last
            if idle_time >= IDLE_UNLOAD_SECONDS:
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

    idle_time = 0
    if last_used:
        idle_time = time.time() - last_used

    return {
        "ok": True,
        "mode": "mock" if MOCK_MODE else "real",
        "loaded": loaded,
        "idle_seconds": idle_time,
        "loaded_at": loaded_at,
        "last_used": last_used,
        "idle_unload_limit": IDLE_UNLOAD_SECONDS,
        "total_requests": total,
    }


@app.post("/admin/unload")
def admin_unload():
    _unload_model()
    return {"ok": True, "unloaded": True}


@app.post("/v1/ocr")
async def ocr(
    file: UploadFile = File(...),
    mode: str = Form("markdown"),
    prompt: Optional[str] = Form(None),
    keep_files: bool = Form(False),
):
    if prompt is None:
        prompt = (
            "<image>\nFree OCR." if mode == "free" else "<image>\nConvert to markdown."
        )

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

        with _infer_lock:
            with _state_lock:
                model = _model
                tok = _tokenizer

            if model is None:
                raise RuntimeError("Model failed to load")

            # In mock mode, we just call the mock method
            if MOCK_MODE:
                text = model.infer()
            else:
                # Real inference
                text = model.infer(
                    tok,
                    prompt=prompt,
                    image_file=img_path,
                    output_path=req_dir,
                    eval_mode=True,
                )

        out_file = os.path.join(req_dir, "result.txt")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(text)

        resp = {"id": req_id, "text": text, "mock": MOCK_MODE}

        if not keep_files:
            try:
                import shutil

                shutil.rmtree(req_dir, ignore_errors=True)
            except:
                pass

        return JSONResponse(resp)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
