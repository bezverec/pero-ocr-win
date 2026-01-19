# Workflow Examples (pero-ocr-win)

This document shows **practical end-to-end workflows** using PERO OCR on Windows
together with the **win-scripts** provided in this repository.

The focus is on:
- reproducibility
- ALTO correctness
- block / region preservation
- conversion to text

---

## Directory Convention

Example working directory:

```text
C:\temp\pero\
├─ in\                    # input images
│  └─ 0001.jpg
├─ out-pagexml\           # PageXML output
├─ out-alto\              # PERO-native ALTO (v2.x)
├─ out-alto-v4_4\         # converted ALTO (v4.4)
├─ out-txt\               # derived TXT files
├─ models\
│  └─ pero_eu_cz_print_newspapers_2022-09-26\
│     └─ config.ini
└─ pero-ocr-win\
   ├─ win-scripts\
   └─ win-requirements.txt
```

---

## Workflow 1: Image → ALTO + PageXML (Primary OCR Output)

This workflow demonstrates **explicit generation of both ALTO and PageXML**
using a single PERO OCR run.

### Command

```powershell
python win-scripts\win-parse_folder.py `
  --config C:\temp\pero\models\pero_eu_cz_print_newspapers_2022-09-26\config.ini `
  --input-image-path C:\temp\pero\in `
  --output-xml-path C:\temp\pero\out-pagexml `
  --output-alto-path C:\temp\pero\out-alto `
  --device gpu
```

### Output

```text
out-pagexml\
└─ 0001.xml          # PageXML

out-alto\
└─ 0001.xml          # ALTO XML (PERO-native, v2.x compatible)
```

### Notes

- `--output-xml-path` → **PageXML**
- `--output-alto-path` → **ALTO**
- Both outputs originate from the same OCR pipeline
- Native ALTO should always be preserved

---

## Workflow 2: ALTO Version Conversion

Convert PERO-native ALTO into a **specific, strictly valid ALTO version**.

### Example: v2.x → v4.4

```powershell
python win-scripts\win-alto_convert.py `
  C:\temp\pero\out-alto\0001.xml `
  C:\temp\pero\out-alto-v4_4\0001.xml `
  --to v4.4 `
  --pretty
```

### Validation Only

```powershell
python win-scripts\win-alto_convert.py `
  C:\temp\pero\out-alto-v4_4\0001.xml `
  NUL `
  --to v4.4 `
  --validate-only
```

---

## Workflow 3: ALTO → TXT (Block-Aware)

Extract text from ALTO **while respecting layout blocks**.

### Command

```powershell
python win-scripts\win-alto_to_txt.py `
  C:\temp\pero\out-alto-v4_4\0001.xml `
  C:\temp\pero\out-txt\0001.txt `
  --paragraphs
```

---

## Workflow 4: PageXML → TXT

Convert PageXML directly to text (layout-aware).

```powershell
python win-scripts\win-pagexml_to_txt.py `
  C:\temp\pero\out-pagexml\0001.xml `
  C:\temp\pero\out-txt\0001.page.txt `
  --paragraphs
```

---

## Workflow 5: Batch Processing

### Convert all PERO ALTO files to ALTO v4.4

```powershell
for %f in (C:\temp\pero\out-alto\*.xml) do (
  python win-scripts\win-alto_convert.py "%f" "C:\temp\pero\out-alto-v4_4\%~nxf" --to v4.4
)
```

### Convert all ALTO v4.4 files to TXT

```powershell
for %f in (C:\temp\pero\out-alto-v4_4\*.xml) do (
  python win-scripts\win-alto_to_txt.py "%f" "C:\temp\pero\out-txt\%~nf.txt" --paragraphs
)
```

---

## Recommended Archival Workflow

1. OCR → **out-alto** (PERO-native, immutable)
2. Convert → **out-alto-v4_4**
3. Validate
4. Derive TXT → **out-txt**
5. Preserve all XML layers

---

## Notes

- Derived formats go to separate directories
- Conversion is deterministic and reversible
- Always keep the **original PERO ALTO** - Conversion is deterministic and reversible - Scripts are Windows-safe (no fcntl, no safe_gpu)
---

Next document:
[**README-upstream.md**](README-upstream.md)


