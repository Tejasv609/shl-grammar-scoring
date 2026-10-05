# Spoken Grammar Scoring — SHL Hiring Assessment 2026

Predict a continuous grammar score (0–5) from ~45–60 second clips of spoken English.
Evaluation: RMSE on a hidden test set.

**Results**

| Metric | Value |
|---|---|
| 5-fold cross-validated RMSE (train) | **0.5363** |
| Cross-validated Pearson r | 0.901 |
| Kaggle public leaderboard | **0.3871** |

`submission.csv` is the scored output (216 rows). `notebook.ipynb` is the full runnable
pipeline that regenerates it — it runs as-is on Kaggle (CPU).

## Approach in one paragraph

Frozen pretrained encoders do the heavy lifting — no fine-tuning. Each audio clip is
transcribed with Whisper; the transcript goes through a frozen DeBERTa text encoder and
the audio goes through three frozen speech encoders (wav2vec2, WavLM, HuBERT). Each
embedding is compressed with PCA-64. A small gradient-boosting ensemble is trained on
the acoustic side, another on the text side, the two predictions are blended
(0.6 acoustic + 0.4 text), and finally predictions are expanded 1.2× around their mean
to undo regression-to-the-mean, then clipped to [0, 5].

## Pipeline

```
Audio → Whisper → transcript ─┬─ spaCy → handcrafted syntactic/prosodic features (67)
                              ├─ DeBERTa-v3 (frozen) → mean-pool → PCA-64
                              └─ Qwen-27B → LLM grammar score (one feature)
Audio ─┬─ wav2vec2-base (frozen) → PCA-64
       ├─ WavLM-large (frozen)   → PCA-64
       └─ HuBERT-large (frozen)  → PCA-64
                                   ▼
        Acoustic: [67 HC + 3×PCA-64 + LLM] → mean(Ridge, RF, 2×LightGBM) ──┐
        Text:     [67 HC + DeBERTa-PCA-64] → mean(Ridge, RF, 2×LightGBM) ─┤
                                                                          ▼
                              0.6 × acoustic + 0.4 × text
                                                                          ▼
                              range expansion ×1.2 around the mean
                                                                          ▼
                                                                  submission.csv
```

## Why each piece is there

- **Frozen embeddings, not fine-tuning.** With only 769 training clips, fine-tuning
  100M+ parameters overfits badly (tried: WavLM 0.76, DeBERTa training instability,
  CoLA-RoBERTa 0.78 vs 0.54 for frozen). Frozen embeddings + gradient boosting is the
  right bias/variance trade-off at this sample size.
- **Three speech encoders, not one.** wav2vec2, WavLM and HuBERT learn different
  representations; PCA-64 each keeps them compact and their errors are complementary.
- **Separate acoustic and text tracks.** Fluency/pronunciation (audio) and
  grammar/vocabulary (transcript) fail on different examples — their prediction
  errors correlate only 0.77, so the 0.6/0.4 blend beats either side alone.
- **LLM grammar score as a feature.** A 27B LLM prompted with a grammar rubric
  scores each transcript 0–5; used as one input feature, it moved the acoustic
  model 0.582 → 0.572.
- **Range expansion ×1.2.** Regression models shrink predictions toward the mean
  (label 2.0 was overpredicted by +0.73, label 5.0 underpredicted by −0.47).
  Expanding predictions 1.2× around their mean restored the true label spread and
  was the single biggest gain: CV 0.552 → 0.536, leaderboard 0.397 → 0.387.

## What was tried and dropped

- T5 grammar-error-correction edit features: correlated 0.19 with labels *in the
  wrong direction* (fluent speakers got "corrected" more) — dropped.
- Character/word TF-IDF: 59k sparse features on 769 samples overfit (CV ~1.0) — dropped.
- Ordinal classification, isotonic calibration, learned stacking, residual
  correction models: all neutral or worse than the plain blend — dropped.

## Reproducing

The notebook runs on Kaggle CPU. It expects three inputs mounted under
`/kaggle/input`:

1. The competition data (`shl-hiring-assessment-2026`)
2. `tejasv002/shl-model-artifacts` — precomputed embeddings, handcrafted features,
   perplexity scores, and text-side predictions
3. `tejasv002/shl-transcripts` — Whisper transcripts (`transcripts.jsonl`)

Executing all cells trains both ensembles and writes `submission.csv`.
The linked Kaggle kernel that produced the scored submission is
[`tejasv002/shl-combined-submission`](https://www.kaggle.com/code/tejasv002/shl-combined-submission).

## Dependencies

See `requirements.txt`: `numpy`, `pandas`, `scikit-learn`, `lightgbm`, `pyarrow`.
