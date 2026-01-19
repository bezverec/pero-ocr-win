# Windows Installation Guide (pero-ocr-win)

This document describes **step-by-step installation of PERO OCR on Windows 11**
using the `pero-ocr-win` fork.  
The guide assumes **Python-only usage (no Docker, no GUI)**.

---

## 1. System Requirements

### Operating system
- Windows 11 (64-bit)

### Hardware
- CPU: x64
- GPU (optional but recommended): NVIDIA GPU with CUDA support  
  (RTX series tested)

### Software
- Python **3.10 or later
- Git
- NVIDIA driver (if GPU is used)

---

## 2. Install Python

1. Download Python from:
   https://www.python.org/downloads/windows/

2. During installation:
   - **Add Python to PATH**
   - Install for all users (recommended)

3. Verify installation:
```powershell
python --version
```

---

## 3. Create Virtual Environment

```powershell
cd C:\temp
mkdir pero
cd pero

python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

---

## 4. Clone Repository

```powershell
git clone https://github.com/bezverec/pero-ocr-win.git
cd pero-ocr-win
```

---

## 5. Install Dependencies (Windows-specific)

```powershell
pip install --upgrade pip setuptools wheel
pip install -r win-requirements.txt
```

This installs:
- PyTorch
- OpenCV
- lxml
- numpy
- other PERO dependencies

---

## 6. Verify Installation

```powershell
python -c "import pero_ocr; print('PERO OCR OK')"
```

---

## 7. Download OCR Model

Download a PERO model (example):

- European printed OCR (Czech newspapers):
  https://nextcloud.fit.vutbr.cz/s/NtAbHTNkZFpapdJ

Example directory layout:
```text
C:\temp\pero\models\
└─ pero_eu_cz_print_newspapers_2022-09-26\
   └─ config.ini
```

---

## 8. Prepare Input / Output Directories

```powershell
mkdir C:\temp\pero\in
mkdir C:\temp\pero\out
```

Put input images into `in` (JPG, PNG, TIFF).

---

## 9. Run OCR (Windows Script)

Use **Windows-compatible script**:

For CPU:
```powershell
python win-scripts\win_parse_folder.py `
  --config C:\temp\pero\models\pero_eu_cz_print_newspapers_2022-09-26\config.ini `
  --input-image-path C:\temp\pero\in `
  --output-xml-path C:\temp\pero\out `
  --device cpu
```

For GPU:
```powershell
python win-scripts\win_parse_folder.py `
  --config C:\temp\pero\models\pero_eu_cz_print_newspapers_2022-09-26\config.ini `
  --input-image-path C:\temp\pero\in `
  --output-xml-path C:\temp\pero\out `
  --device gpu `
  --gpu-id 0
```

---

## 10. Outputs

For each image:
- `*.xml` – PageXML (default)

Further processing is described in:
- `alto-versions.md`
- `workflow-examples.md`

---

## Notes

- Windows uses **win-scripts/** instead of `user_scripts/`
- No `safe_gpu`, no `fcntl`
- Upstream scripts remain untouched

---

Next document:
[**gpu-cuda-pytorch.md**](gpu-cuda-pytorch.md)


