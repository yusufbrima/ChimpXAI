# 🧾 Revision & Re-analysis To-Do

### Project: *Chimpanzee Individual Recognition (Pant-hoot Classification)*

**Purpose:** Rerun experiments after correcting label issues and fully address reviewer & editor comments.

---

## 🧠 Main Objectives

1. **Rebuild dataset & splits**

   * [ ] Audit and correct all labels → save as `labels_v2.csv`.
   * [ ] Document corrections (old label, new label, corrected by, reason).
   * [ ] Generate summary table: per-individual counts before/after correction.
   * [ ] Ensure no data leakage: use *GroupKFold* based on recording/session IDs.
   * [ ] Save consistent train/val/test splits (JSON or CSV).

2. **Data preprocessing**

   * [ ] Standardize mel-spectrogram extraction (SR, FFT, hop, mel bins, etc.).
   * [ ] Implement optional *chimpanzee-hearing-scale* filterbank for ablation.
   * [ ] Compute call duration stats (mean ± SD).
   * [ ] Justify 2-second window length (distribution plot).

3. **Augmentation & balancing**

   * [ ] Build robust augmentation pipeline (noise, pitch, stretch, specAugment).
   * [ ] Add class-balanced sampling / oversampling for minority individuals.
   * [ ] Save augmented sample logs (to reproduce later).
   * [ ] Test augmentation-only baseline (to check over-augmentation).

4. **Model suite**

   * [ ] CNN baseline (previous architecture).
   * [ ] ViT (Vision Transformer on spectrograms).
   * [ ] AST (Audio Spectrogram Transformer).
   * [ ] Optional: attention-CNN or CRNN baseline for comparison.
   * [ ] Verify all models take same input dimensions & preprocessing.

5. **Hyperparameter tuning (Optuna)**

   * [ ] Define search spaces for each model type.
   * [ ] Run 50–100 trials/model with pruning.
   * [ ] Save study DBs and top-5 parameter sets.
   * [ ] Export best params to `configs/best_*.json`.

6. **Final training runs**

   * [ ] Train with best params on full train+val set.
   * [ ] Run 3 random seeds for each model.
   * [ ] Save checkpoints, logs, and metrics.
   * [ ] Evaluate on held-out test set.
   * [ ] Compute per-individual confusion matrices.

7. **Evaluation metrics**

   * [ ] Accuracy, balanced accuracy, macro-F1.
   * [ ] ROC/AUC per individual (optional).
   * [ ] Significance tests (paired t/Wilcoxon).
   * [ ] Benchmark vs earlier results (table update).

8. **Explainability & visualization**

   * [ ] Implement Score-CAM (CNN) → specify conv layer.
   * [ ] Implement attention visualization (ViT, AST).
   * [ ] Quantify CAM overlap with annotated call regions (% + IoU).
   * [ ] Add noise/de-noise test to verify non-background focus.
   * [ ] Use same example calls for all Figures 3–5.
   * [ ] Label y-axis in Hz, improve resolution, add originals.

9. **Documentation & reproducibility**

   * [ ] Record environment (requirements.txt / conda YAML).
   * [ ] Fix random seeds (Python, NumPy, Torch).
   * [ ] Save Optuna logs and metrics in `results/`.
   * [ ] Update README + scripts reference.
   * [ ] Produce pipeline diagram (optional for supplement).

10. **Manuscript integration**

* [ ] Update Methods (dataset correction, augmentation, tuning).
* [ ] Expand interpretability description (Score-CAM, ViT attention).
* [ ] Revise Table 2 (include DeepSqueak, primate ML studies).
* [ ] Replace “high/robust” wording with cautious phrasing.
* [ ] Move anatomy discussion to Discussion.
* [ ] Add dataset context (individual count, group, age/rank info).
* [ ] Insert new confusion matrix (Fig 2).
* [ ] Regenerate Figs 3–5 with new spectrograms + CAM overlays.
* [ ] Check all axis units and DPI before submission.

11. **Final checks before resubmission**

* [ ] Confirm all reviewer/editor points addressed.
* [ ] Prepare response-to-reviewers table mapping actions.
* [ ] Verify reproducibility (one-command re-run).
* [ ] Run spell/grammar and clarity check.
* [ ] Update references & benchmarks.
* [ ] Export final figures and tables.

---

## 🗓 Suggested Timeline

| Week | Focus                        | Key Deliverables                   |
| ---- | ---------------------------- | ---------------------------------- |
| 1    | Label audit & preprocessing  | Corrected dataset, call stats      |
| 2–3  | Augmentation + Optuna tuning | CNN + ViT tuned                    |
| 4    | AST tuning + final models    | AST trained, best configs          |
| 5    | Explainability analyses      | CAM overlap, noise tests, figures  |
| 6    | Manuscript revision          | Updated text, tables, response doc |

---

## 🧩 Quick Command Reminders

```bash
# run preprocessing
python src/preprocessing.py --input data/raw --output data/processed

# tune model
python src/optuna_objective.py --model cnn --trials 100

# train final model
python src/train_script.py --model vit --config configs/best_vit.json

# evaluate CAM overlap
python src/evaluation.py --cam-overlap --threshold 0.5
```

---

## 🧠 Notes / Ideas to Revisit

* [ ] Try Mixup/CutMix augmentation on spectrograms.
* [ ] Test smaller ViT (tiny or base patch-16) to avoid overfitting.
* [ ] Consider self-supervised pretraining (BYOL-A or AudioMAE) if time allows.
* [ ] Compare CAM overlap across models as quantitative interpretability metric.
* [ ] Evaluate “chimp-hearing filterbank” improvement effect.