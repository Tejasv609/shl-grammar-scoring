# SHL Spoken Grammar Scoring — Project Journey (Step by Step)

**Task:** Predict a continuous grammar score (0–5) from ~45–60 s clips of spoken English.
**Metric:** RMSE. **Data:** 769 labeled train clips, 216 test clips, 985 total transcripts.
**Final result:** 5-fold CV RMSE **0.5363** (Pearson r 0.901), Kaggle public LB **0.3871**.

This document reconstructs every major step in order, with the numbers that drove
each decision. Failed lanes are included — they explain why the final model looks
the way it does.

---

## Step 0 — Understand the problem

Labels are ordinal-ish (0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0) but scored
with RMSE, so regression is the natural framing. Two information sources exist in
each clip: **how it sounds** (fluency, pronunciation, prosody) and **what was said**
(grammar, vocabulary). The final model exploits both.

Key data facts found early (EDA):
- Label distribution is imbalanced: 37 samples at 0.0, only 1 at 1.0, 102 at 2.0, 133 at 5.0.
- Word count correlates 0.286 with the label — longer answers score higher.
- No duplicates, no empty transcripts.
- Adversarial validation (train vs test classifier) reached AUC **0.73** — there IS a
  real train/test distribution shift (test clips have lower voiced ratio, higher LLM
  scores). This explains why local CV (0.536) and leaderboard (0.387) differ in scale.

## Step 1 — First baseline: wav2vec2 + handcrafted features

- Transcribed everything with faster-whisper (`base.en`) → `transcripts.jsonl`.
- Extracted wav2vec2-base embeddings (mean-pooled, 768-dim) + 67 handcrafted
  features (spaCy syntax + librosa prosody).
- Result: **OOF 0.6919, LB 0.5627**. A working baseline, but far from competitive.

## Step 2 — Add WavLM-large: the first real jump

- Added frozen WavLM-large embeddings, PCA-compressed to 64 dims.
- Handcrafted + WavLM: **OOF 0.5827, LB 0.4508 (rank 13)**. Leader was 0.3392.
- Lesson: bigger frozen speech encoders help a lot; PCA-64 keeps them compact.

## Step 3 — Combine all frozen speech embeddings → plateau

- Added HuBERT-large (extracted on Colab GPU, 985 × 2048-dim vectors).
- wav2vec2 + WavLM + HuBERT combined: **OOF 0.5821** — no better than WavLM alone.
- Lesson: the three encoders overlap heavily. More of the same representation
  doesn't add signal. (This plateau later motivated the *text* track.)

## Step 4 — Build the text track: transcripts as a second view

- Embedded Whisper transcripts with frozen DeBERTa-v3-base (mean-pool, 768-dim),
  PCA-64, trained a 4-model ensemble (Ridge + RF + 2× LightGBM).
- Text alone: **OOF 0.7410** — weak by itself (grammar signal is noisy in ASR text).
- Handcrafted + text: **OOF 0.5925**.
- **Blend 0.5 × acoustic + 0.5 × text: OOF 0.5526** — the best validation yet.
- Submitted → **LB 0.4005 (rank 21)**. Then tuned blend weights → **LB 0.3974**,
  hitting the sub-0.4 target.

Why it worked: the acoustic and text models make *different* errors (error
correlation ~0.77). Fluency problems show in audio; grammar problems show in text.
Averaging two views that fail differently is the oldest ensembling win there is.

## Step 5 — LLM grammar scores as a feature

- Scored all 985 transcripts 0–5 with Qwen-27B prompted with a grammar rubric
  (Groq API). LLM-vs-label correlation: **0.58**.
- As a single feature on the acoustic side: **0.5821 → 0.5719** (helped).
- But it killed ensemble diversity (error correlation with text side rose
  0.77 → 0.81), so the full blend barely moved: **0.5526 → 0.5530**.
- Best grid point 0.5520 was submitted anyway → **LB 0.3979**.
- Lesson: a feature can help one model while hurting the ensemble. Always judge
  by the *final* blend, not the component.

## Step 6 — THE BREAKTHROUGH: range expansion ×1.2

Error analysis showed the model regressed to the mean: label 2.0 was overpredicted
by **+0.73**, label 5.0 underpredicted by **−0.47**. Regression models shrink
predictions toward the training mean — that's structural, not fixable with more
features.

Fix: expand predictions 1.2× around their mean, clip to [0, 5].

```python
final = clip(mean + (blend - mean) * 1.2, 0, 5)
```

- OOF: **0.5520 → 0.5363** (nested-CV verified, stable across folds)
- LB: **0.3974 → 0.3871** ← the champion score
- This single one-line change beat every feature-engineering idea tried before
  or since.

## Step 7 — Fine-tuning attempts (all failed, all closed)

With 769 samples, fine-tuning 100M+ parameters was always a long shot. Tried anyway:

| Attempt | Result | Verdict |
|---|---|---|
| WavLM fine-tune (Colab) | fold RMSE ~2.22, disconnected | broken recipe (head LR too small) |
| WavLM fixed recipe (Kaggle GPU) | OOF **0.7641** | overfits; closed |
| DeBERTa fine-tune (15 kernel versions!) | NaN gradients — AdamW corrupts DeBERTa params on CUDA | closed |
| DeBERTa head-only SGD workaround | OOF **0.9431** | closed |
| CoLA-RoBERTa fine-tune | OOF **0.7819** | closed |

Lesson: frozen embeddings + gradient boosting is the right bias/variance trade-off
at n=769. Fine-tuning is not "more powerful" here — it's just more overfitting.

## Step 8 — The "smart ideas" graveyard

Every one of these was tested with proper 5-fold CV against the 0.5363 champion.
All neutral or worse:

| Idea | OOF | Verdict |
|---|---|---|
| Ordinal cumulative classifiers P(y≥k) | 0.6073 (0.6207 expanded) | worse |
| Isotonic calibration | 0.5589 | worse |
| Targeted LLM residual correction (label-2.0) | 0.5388 | worse |
| Raw LLM as direct predictor | 1.2383 | terrible — LLM is a *feature*, not a predictor |
| PCA-32 instead of PCA-64 | 0.5400 | worse |
| Inverse-frequency sample weighting | 0.5373 | neutral |
| Error-based sample weighting | 0.5491 | worse |
| Adversarial importance weighting (for the train/test shift) | 0.5409 | neutral — shift is real but unfixable this way |
| Learned stacking weights | 0.5424 | worse than fixed 0.6/0.4 |
| 80 deep syntactic features (spaCy v3) | 0.5847 (+0.019 worse) | closed |
| Prosody v2 features | 0.5701 | worse |
| Whisper word-confidence features | no gain | signal already in embeddings |
| T5 grammar-error-correction edit features | 0.6058 (corr 0.19 *wrong direction*) | closed |
| Character + word TF-IDF (59k features) | **1.0412** | severe overfitting; closed |
| Handcrafted-only (best: GBR) | 0.7643 | embeddings are essential |

## Step 9 — The residual / grammar-model phase

After the representation was declared exhausted, three targeted attempts:

1. **Residual LightGBM** (predict `label − champion_pred` from all 260 features):
   0.5376 → **0.5677**. Circular — the residual is noise given the same features.
2. **Low-grammar specialist** (binary y≤2, weighted toward hard cases): AUC 0.8745
   overall, but mean predicted probability on the actual hard blind-spot cases
   was **0.010**. Correction worsened RMSE → 0.5524.
3. **GEC embedding delta** (DeBERTa distance between original and T5-corrected
   transcript): correlation **+0.039** with label — measures stylistic rewriting,
   not grammaticality.

Verdict: the 260-feature representation is fully squeezed. The remaining gap to
#1 (0.3278) needs a genuinely new representation (e.g., GPU fine-tuning of a
small language model on transcripts), not more cleverness on the same features.

## Step 10 — The champion (what was submitted)

```
Audio → Whisper → transcript ─┬─ spaCy → 67 handcrafted ──────────────┐
                              ├─ DeBERTa-v3 (frozen) → PCA-64 ─────────┤
                              └─ Qwen-27B → LLM grammar score ─────────┤
Audio ─┬─ wav2vec2-base (frozen) → PCA-64 ─────────────────────────────┤
       ├─ WavLM-large (frozen)   → PCA-64 ─────────────────────────────┤
       └─ HuBERT-large (frozen)  → PCA-64 ─────────────────────────────┤
                                                                       ▼
        Acoustic: [HC + 3×PCA-64 + LLM] → mean(Ridge, RF, 2×LGBM) ──┐
        Text:     [HC + DeBERTa-PCA-64] → mean(Ridge, RF, 2×LGBM) ─┤
                                                                    ▼
                        0.6 × acoustic + 0.4 × text
                                                                    ▼
                        ×1.2 range expansion around the mean
                                                                    ▼
                                                             submission.csv
```

| Metric | Value |
|---|---|
| 5-fold CV RMSE | **0.5363** |
| CV Pearson r | 0.901 |
| In-sample training RMSE | 0.248 |
| Kaggle public LB | **0.3871** |

Submitted as one linked Kaggle artifact: notebook `tejasv002/shl-combined-submission`
regenerates `submission_llm_expanded.csv`. GitHub: `Tejasv609/shl-grammar-scoring`.

## Step 11 — What I'd do next (honest)

1. **LoRA fine-tune a small LM** (DistilBERT/MiniLM) on transcripts only — needs GPU.
   Public solutions report this works; our fine-tuning failures were bad recipes,
   not proof the idea is dead.
2. **Rubric-conditioned training** — use the scoring rubric as training signal
   (predict P(score) per rubric level, take expectation).
3. Nothing else on the current features. They're done.
