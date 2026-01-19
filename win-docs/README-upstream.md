# README – Upstream Relation (pero-ocr-win)

This repository **pero-ocr-win** is a **Windows-focused fork** of the original
PERO OCR project developed at Brno University of Technology.

Its purpose is **not to replace** the upstream project, but to provide:

- a Windows-friendly experience
- reproducible installation on Windows 11
- CUDA / PyTorch guidance for Windows
- strictly valid ALTO conversion workflows
- helper scripts without POSIX-only dependencies

---

## Upstream Project

Original repository:

- https://github.com/DCGM/pero-ocr

Authors:
- DCGM / FIT BUT (Brno University of Technology)

License:
- See upstream LICENSE file (preserved in this fork)

---

## What This Fork Changes

### 1. Windows Compatibility

The upstream repository contains Unix/Linux assumptions:

- `safe_gpu`
- `fcntl`
- shell-oriented workflows

This fork avoids patching upstream code and instead provides:

- Windows-safe replacement scripts
- isolated `win-scripts/` directory
- no changes to core OCR logic

---

### 2. Script Strategy (No Patching)

**Upstream scripts are not modified.**

Instead:

| Purpose | Script |
|------|-------|
| OCR runner | `win-parse_folder.py` |
| ALTO conversion | `win-alto_convert.py` |
| ALTO → TXT | `win-alto_to_txt.py` |
| PageXML → TXT | `win-pagexml_to_txt.py` |

This guarantees:
- clean rebase on upstream
- no merge conflicts
- full upstream traceability

---

### 3. ALTO Version Support

Upstream PERO OCR outputs **ALTO v2.x–compatible XML**.

This fork adds:
- strict conversion between ALTO versions
- validation for:
  - v2.0
  - v2.1
  - v3.0
  - v3.1
  - v4.0
  - v4.1
  - v4.2
  - v4.3
  - v4.4

No OCR content is modified or fabricated.

---

### 4. What This Fork Does NOT Do

This repository deliberately does **not**:

- change OCR models
- alter neural network behavior
- invent semantic structure
- add GUI (yet)
- reimplement PERO internals

All OCR intelligence remains upstream.

---

## Keeping in Sync with Upstream

Recommended workflow:

```bash
git remote add upstream https://github.com/DCGM/pero-ocr.git
git fetch upstream
git rebase upstream/master
```

Windows-specific files live in:
- `win-scripts/`
- `win-requirements.txt`
- `docs/`

These files are expected to remain conflict-free.

---

## When to Use Upstream Directly

Use upstream `pero-ocr` if:

- you are on Linux
- you use Docker exclusively
- you rely on `safe_gpu`
- you do not need ALTO conversions

---

## When to Use This Fork

Use `pero-ocr-win` if:

- you work on Windows
- you need CUDA + PyTorch on Windows
- you need strict ALTO validation
- you integrate with digital libraries
- you require reproducible OCR pipelines

---

## Attribution

If you publish work using this fork, please cite:

> PERO OCR – Document content extraction and recognition  
> DCGM, Faculty of Information Technology, BUT

And optionally mention:
> Windows integration and ALTO tooling provided by pero-ocr-win fork

---

## Final Note

This fork exists to **lower friction**, not to fragment the ecosystem.

All improvements are designed to be:
- transparent
- reversible
- upstream-friendly

---

End of document.
