# MedGemma + AnyJev: NHATS Clock Drawing Evaluation

## Objective
Adapt AnyJev's label-free L0 calibration to MedGemma 4B for
ordinal Clock Drawing Test scoring and compare performance
against existing NHATS baselines.

## Method
MedGemma processes a clock drawing image and Prompt C.
Restricted next-token softmax produces probabilities for
the six NHATS scores (0 through 5).

We estimate a fixed label prior from 200 separate,
patient-disjoint NHATS test images and apply:

P_L0(k) = normalize(P_raw(k) / prior(k)^0.75)

The implementation reuses AnyJev's batch_prior and
apply_contextual functions. Unlike the original running
batch-prior Decider, this adaptation uses a fixed prior.
AnyJev explicitly skips option permutations for ordered
score questions, so no permutation step is applied here.

## Dataset
- Source: Jiaqing's authorized nhats_cropped/test directory
- Calibration: 200 drawings
- Evaluation: 273 matched drawings from the permitted test set
- Shared patients between groups: 0
- Shared images between groups: 0
- Ground-truth scores: NHATS ordinal scale 0 through 5

## Results

| Model | Accuracy | Balanced accuracy | MAE |
|---|---:|---:|---:|
| MedGemma + L0 | 37.00% | 25.55% | 0.7839 |
| Raw MedGemma | 36.26% | 24.28% | 0.8095 |
| GPT-5 | 31.50% | 34.73% | 0.9560 |
| XGBoost | 29.30% | 27.78% | 1.2894 |
| Mini-VGG | 22.71% | 16.67% | 1.1209 |

## Findings
L0 improved accuracy by approximately 0.73 percentage points
and reduced MAE by 0.0256.

Of 273 predictions, 13 changed after calibration:
10 improved and 3 worsened in absolute error.

MedGemma continued to overpredict score 4:
214 raw predictions and 202 L0 predictions, compared
with 81 ground-truth score-4 drawings.

GPT-5 achieved higher balanced accuracy than MedGemma.

## Verification
- 273 prediction rows aligned with baseline records
- Saved L0 probabilities independently reproduced
- Real MedGemma inference reproduced saved probabilities exactly
  for one image
- Real GPU end-to-end smoke test passed
- 131 automated tests passed; 4 environment/optional tests skipped

## Limitations
This is a matched-subset evaluation, not a full-test-set benchmark.
Results are exploratory. Prompt development involved earlier
NHATS test examples, and the matched subset may not represent
the full NHATS population. Calibration improvement has not
been established as statistically significant.

Baseline preprocessing may differ across models.
The fixed-prior multimodal adapter is not a drop-in
replacement for AnyJev's original Decider.

Private NHATS images and patient-level predictions must
not be included in the public repository.
