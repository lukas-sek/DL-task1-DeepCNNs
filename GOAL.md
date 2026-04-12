# GOAL — Practical Work 1: Deep CNNs on MAMe

## RALPH Loop Protocol

**R**ead → **A**nalyze → **L**ist gaps → **P**lan next → **H**andoff (submit job)

### How to trigger a loop iteration
After a BSC job finishes and you have downloaded `metrics.json`:
> "RALPH: std_underfit results are in"

Claude will:
1. Read `runs/<name>/metrics.json`
2. Check done criteria below
3. Write/update the relevant `report/` section draft
4. Mark status in this file
5. Output the next recommended launcher config

---

## Report Structure & Section → Experiment Mapping

| # | Report Section | Pages | Experiment(s) needed | Status |
|---|---------------|-------|----------------------|--------|
| 1 | Introduction + Dataset | 0.5 | — (write from spec) | ⬜ TODO |
| 2 | Data Preprocessing | 0.5 | — (write from code) | ⬜ TODO |
| 3 | Standard CNN | 2.0 | std_underfit, std_overfit, std_best | ⬜ TODO |
| 4 | Non-standard CNN | 2.0 | res_underfit, res_overfit, res_best | ⬜ TODO |
| 5 | Transfer Learning (optional) | 1.0 | transfer_frozen, transfer_finetune | ⬜ TODO |
| 6 | Comparison Table | 0.5 | all runs complete | ⬜ TODO |
| 7 | Discussion & Conclusions | 1.0 | all runs complete | ⬜ TODO |
| 8 | LLM Declaration | ~0.1 | — | ⬜ TODO |
| 9 | References | ~0.4 | — | ⬜ TODO |

---

## Experiment Registry

| Experiment | Folder | Status | val_acc | test_acc | Notes |
|-----------|--------|--------|---------|----------|-------|
| std_underfit | runs/std_underfit | ⬜ NOT RUN | — | — | |
| std_overfit | runs/std_overfit | ⬜ NOT RUN | — | — | |
| std_best | runs/std_best | ⬜ NOT RUN | — | — | |
| res_underfit | runs/res_underfit | ⬜ NOT RUN | — | — | |
| res_overfit | runs/res_overfit | ⬜ NOT RUN | — | — | |
| res_best | runs/res_best | ⬜ NOT RUN | — | — | |
| transfer_frozen | runs/transfer_frozen | ⬜ OPTIONAL | — | — | |
| transfer_finetune | runs/transfer_finetune | ⬜ OPTIONAL | — | — | |

---

## Done Criteria (per experiment)

### std_underfit — DONE when:
- [ ] `metrics.json` present
- [ ] `final_train_acc` < 45%
- [ ] `final_val_acc` within 5% of `final_train_acc` (both low → underfitting confirmed)
- [ ] Loss curve plot saved to `runs/std_underfit/plots/`
- [ ] `report/03_standard_cnn.md` underfitting paragraph written

### std_overfit — DONE when:
- [ ] `metrics.json` present
- [ ] `final_train_acc` > 80%
- [ ] `final_val_acc` at least 15% below `final_train_acc` (gap → overfitting confirmed)
- [ ] Loss curve plot saved
- [ ] `report/03_standard_cnn.md` overfitting paragraph written

### std_best — DONE when:
- [ ] `metrics.json` present
- [ ] `final_val_acc` > 45% (meaningful generalisation on 29 classes)
- [ ] Train/val gap < 10%
- [ ] All plots saved (loss, accuracy, confusion matrix)
- [ ] `report/03_standard_cnn.md` regularization + results paragraph written

### res_underfit / res_overfit / res_best — same criteria as std above

### Assignment DONE when:
- [ ] std_underfit ✅
- [ ] std_overfit ✅
- [ ] std_best ✅
- [ ] res_underfit ✅
- [ ] res_overfit ✅
- [ ] res_best ✅
- [ ] All report sections drafted (report/*.md)
- [ ] All figures in figures/ with matching data files (CSV/JSON)
- [ ] Comparison table complete (≥2 models with val+test acc)
- [ ] Total report ≤ 9 pages when compiled

---

## Numeric Data Policy
Every figure in the report must have a matching data file:
- `figures/<name>.pdf` → `figures/<name>.json` (or `.csv`)
- Loss/accuracy curves: JSON with `{epoch, train_loss, val_loss, train_acc, val_acc}`
- Confusion matrix: JSON with `{labels, matrix}`
- Comparison bar chart: CSV with model, val_acc, test_acc, params

---

## metrics.json Schema (written by train.py after each run)
```json
{
  "experiment": "std_underfit",
  "config": {
    "model": "standard_small",
    "input_size": 128,
    "epochs": 30,
    "batch_size": 64,
    "optimizer": "adam",
    "lr": 0.001,
    "weight_decay": 0.0,
    "dropout": 0.0,
    "batchnorm": false,
    "augment": false
  },
  "history": {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": []
  },
  "final_train_acc": 0.0,
  "final_val_acc": 0.0,
  "best_val_acc": 0.0,
  "best_epoch": 0,
  "test_acc": null,
  "num_params": 0,
  "time_per_epoch_s": 0.0,
  "total_time_s": 0.0
}
```
