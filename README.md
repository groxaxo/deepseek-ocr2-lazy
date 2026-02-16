# DeepSeek-OCR-2 Lazy-Loading Server

A **Level 1 lazy-loading "instance"**: a small **FastAPI server** that **does not load DeepSeek-OCR-2 until the first request**, then **auto-unloads after an idle timeout** (freeing most VRAM), while the process stays up.

This uses the **Unsloth DeepSeek-OCR-2 path** (`snapshot_download` + `FastVisionModel.from_pretrained`) from Unsloth's guide.

## 1) Install (conda, safe against torch overrides)

```bash
# 1) Create env (Python 3.11 is fine)
conda create -n deepseek-ocr2-unsloth python=3.11 -y
conda activate deepseek-ocr2-unsloth

# 2) Install PyTorch your usual way (pick ONE):
# If you're on CUDA 12.1:
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# If you're on CUDA 11.8:
# conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 3) Install Unsloth without letting pip replace torch (Unsloth recommends upgrading; this is the "no-deps" safe variant)
pip install --upgrade pip
pip install --upgrade --force-reinstall --no-deps --no-cache-dir unsloth unsloth_zoo

# 4) Server deps
pip install fastapi uvicorn python-multipart pillow huggingface_hub
```

Unsloth supports multiple torch/CUDA combos and provides install guidance (incl. Ampere-specific variants).

## 2) Configuration

The server is configured via environment variables. Defaults are set for a typical 15-minute idle unload.

| Variable | Default | Description |
|----------|---------|-------------|
| `DS_OCR2_MODEL_ID` | `unsloth/DeepSeek-OCR-2` | HuggingFace model ID |
| `DS_OCR2_SNAPSHOT_DIR` | `~/models/deepseek_ocr2_unsloth` | Local cache directory for model weights |
| `DS_OCR2_IDLE_UNLOAD_SECONDS` | `900` | Seconds before unloading model (default 15 mins) |
| `DS_OCR2_LOAD_IN_4BIT` | `0` | Set to `1` or `true` to use 4-bit quantization (saves VRAM) |
| `DS_OCR2_PORT` | `8012` | Server port |

## 3) Run it

```bash
conda activate deepseek-ocr2-unsloth
python deepseek_ocr2_lazy_server.py
```

First OCR request will be the "cold load" (download/compile/load). After that it stays warm until the idle timeout is reached.

## 4) Test (curl)

**Markdown extraction (default):**

```bash
curl -s -X POST "http://127.0.0.1:8012/v1/ocr" \
  -F "file=@/path/to/your_image.jpg" \
  -F "mode=markdown"
```

**Plain OCR:**

```bash
curl -s -X POST "http://127.0.0.1:8012/v1/ocr" \
  -F "file=@/path/to/your_image.jpg" \
  -F "mode=free"
```

## 5) Systemd (optional, keeps it running)

Create: `~/.config/systemd/user/deepseek-ocr2-lazy.service`

```ini
[Unit]
Description=DeepSeek-OCR-2 (Unsloth) Lazy Level 1
After=network-online.target

[Service]
Type=simple
Environment=CUDA_VISIBLE_DEVICES=0
Environment=DS_OCR2_PORT=8012
Environment=DS_OCR2_IDLE_UNLOAD_SECONDS=900
Environment=DS_OCR2_SNAPSHOT_DIR=%h/models/deepseek_ocr2_unsloth
WorkingDirectory=%h
ExecStart=/bin/bash -lc 'conda run --no-capture-output -n deepseek-ocr2-unsloth python %h/deepseek-ocr2-lazy/deepseek_ocr2_lazy_server.py'
Restart=always
RestartSec=2

[Install]
WantedBy=default.target
```

*Note: Adjust the `ExecStart` path if your cloned folder path differs.*

Enable + start:

```bash
systemctl --user daemon-reload
systemctl --user enable --now deepseek-ocr2-lazy.service
journalctl --user -u deepseek-ocr2-lazy.service -f
```

## Credits

- **DeepSeek-OCR-2** by DeepSeek AI
- **Unsloth** for the optimization and inference integration
