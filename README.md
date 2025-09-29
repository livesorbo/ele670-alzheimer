# ELE670 Alzheimer’s Classification — Starter

Dette er et *minimalt* skjelett som følger oppgaven din:
- **Baseline:** 2D CNN på enkelt-slices
- **Multi-slice:** stakk 3 nabo-slices som 3 kanaler
- **Subject-level aggregation:** gjennomsnitt / majority vote per subject

## Mappestruktur
```
ele670_project/
├─ data/
│  ├─ train.csv
│  ├─ val.csv
│  └─ test.csv
├─ results/        # lagres: beste modell, logger
├─ src/
│  ├─ dataset.py
│  ├─ model.py
│  ├─ train.py
│  └─ utils.py
└─ main.py
```

## 1) Installer avhengigheter
Opprett evt. et venv og installer:
```bash
pip install -r requirements.txt
```

## 2) Forbered data (CSV)
Legg NIfTI-stier og labels (0=kontroll, 1=AD) i CSV-filene:
- `data/train.csv`
- `data/val.csv`
- `data/test.csv`

Format (semikolon eller komma fungerer – vi bruker komma):
```
subject_id,nifti_path,label
OAS1_0001,data/OAS1_0001.nii.gz,0
OAS1_0002,data/OAS1_0002.nii.gz,1
```

> **Tips:** Start med noen få eksempler for å verifisere at pipeline virker, og utvid etterhvert.

## 3) Kjør baseline-trening (CPU er OK)
```bash
python main.py --train_csv data/train.csv --val_csv data/val.csv --test_csv data/test.csv \
  --epochs 3 --batch_size 8 --lr 1e-3 --multi_slice False
```

## 4) Kjør med multi-slice (3 kanaler)
```bash
python main.py --train_csv data/train.csv --val_csv data/val.csv --test_csv data/test.csv \
  --epochs 3 --batch_size 8 --lr 1e-3 --multi_slice True
```

## 5) Resultater
- Slice-nivå: accuracy/AUC logges under trening.
- Subject-nivå: beregnes etter val/test med gjennomsnitt av slice-probabiliteter per subject.
  Fil `results/metrics.json` oppdateres.

## Notater
- Normalisering: z-score per slice.
- Sliceseleksjon: sentrale 60% av slices (mindre bakgrunn).
- Resampling til 1mm³ kan legges til senere om nødvendig.

Lykke til! 🚀
