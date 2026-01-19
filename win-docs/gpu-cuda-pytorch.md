# GPU, CUDA and PyTorch on Windows (Pero OCR)

This document explains how to correctly set up **CUDA + PyTorch** on **Windows 11**
for use with **Pero OCR**, including common pitfalls and verification steps.

---

## Supported setup (recommended)

Tested and recommended configuration:

- Windows 11 (x64)
- NVIDIA GPU (RTX series or newer)
- NVIDIA driver ≥ 535
- CUDA Toolkit **12.x**
- PyTorch **2.x (CUDA 12.x build)**
- Python **3.10 or later**
- Virtual environment (`venv`)

> ⚠️ Do **not** use system Python without a virtual environment.

---

## 1. Install NVIDIA driver

1. Download the **latest NVIDIA driver**:
   https://www.nvidia.com/Download/index.aspx

2. During installation:
   - Choose **Custom (Advanced)**
   - Enable **Clean installation**

3. Verify installation:

```powershell
nvidia-smi
```

Expected output:
- GPU name
- Driver version
- CUDA version (reported by driver)

---

## 2. Install CUDA Toolkit (optional)

PyTorch wheels already include CUDA runtime,  
but installing CUDA Toolkit helps with debugging and compatibility.

After installation, verify:

```powershell
nvcc --version
```

If `nvcc` is not found, ensure:
- `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin`
  is in `PATH`

---

## 3. Create Python virtual environment

From repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Upgrade base tools:

```powershell
python -m pip install --upgrade pip setuptools wheel
```

---

## 4. Install PyTorch with CUDA support

⚠️ **Do not use default PyPI torch wheels**  
They may install CPU-only builds.

Use **official PyTorch CUDA index**:

```powershell
pip install torch torchvision torchaudio ^
  --index-url https://download.pytorch.org/whl/cu129
```

(You may replace `cu129` with different version if needed.)

---

## 5. Verify PyTorch GPU availability

Run Python REPL:

```powershell
python
```

Then:

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
```

Expected output:

- `True` for `cuda.is_available()`
- Correct GPU name (e.g. RTX 4070, RTX 5070, etc.)

If `False`:
- Driver mismatch
- Wrong PyTorch wheel
- Old GPU / unsupported compute capability

---

## 6. Install Pero OCR dependencies

Install Windows-specific dependencies:

```powershell
pip install -r win-requirements.txt
```

This includes:
- PyTorch (already installed)
- OpenCV
- NumPy
- lxml
- Shapely
- Pillow
- Other OCR-related libraries

---

## 7. Run Pero OCR with GPU

Example (Windows-safe script):

```powershell
python win-scripts\win_parse_folder.py `
  --config C:\models\pero\config.ini `
  --input-image-path C:\data\images `
  --output-xml-path C:\data\alto `
  --device gpu
```

If GPU is correctly detected, logs will include:
- CUDA device usage
- Faster processing times
- No fallback to CPU

---

## 8. Common problems and solutions

### `torch.cuda.is_available() == False`

- Wrong torch build → reinstall with `cu12x`
- NVIDIA driver too old
- Running outside venv

### `RuntimeError: CUDA error`

- Driver / CUDA mismatch
- Reboot after driver install
- Try newer PyTorch build

### Very slow inference

- Running on CPU (`--device cpu`)
- GPU VRAM exhausted → reduce batch size or resolution

---

## 9. Notes for reproducibility

- Always pin PyTorch version for production
- Keep driver updated on Windows
- Avoid mixing CUDA versions in PATH
- Prefer **one GPU per process**

---

## Summary

✔ NVIDIA driver installed  
✔ PyTorch CUDA wheel installed  
✔ `torch.cuda.is_available()` returns `True`  
✔ Pero OCR runs with `--device gpu`

You now have a **fully functional GPU OCR setup on Windows**.

---

