# MedGemma + AnyJev for NHATS Clock Drawing Scoring

## Overview

This project adapts AnyJev's L0 contextual calibration to multimodal
MedGemma 4B for ordinal clinical Clock Drawing Test (CDT) scoring.

MedGemma receives a clock drawing image and a scoring prompt and produces
a probability distribution over NHATS scores 0–5.

## Implementation

- `anyjev/backends/medgemma.py`: MedGemma image-text inference backend.
- `anyjev/medgemma_decider.py`: Fixed-prior L0 calibration adapter.
- `tests/test_medgemma_backend.py`: Backend token-validation tests.
- `tests/test_medgemma_decider.py`: Calibration and persistence tests.

The backend extracts next-token logits for six single-token score labels
and normalizes their probabilities over those six alternatives.

The L0 correction is:

    P_L0(y|x) ∝ P_raw(y|x) / P_prior(y)^0.75

The fixed prior is estimated from a separate calibration sample.

This is a multimodal adapter using AnyJev calibration functions, not a
drop-in replacement for the original text-only AnyJev Decider.

## Dataset

NHATS Clock Drawing Test images with ordinal scores 0–5.

Matched evaluation:
- 273 unique test drawings.
- Same drawings used to compare GPT-5, XGBoost, Mini-VGG, and MedGemma.
- Corrected ground-truth labels, including Round 10 inconsistencies.
- 200 calibration drawings with no patient overlap with the matched evaluation.

Dataset files and images are not included in the repository.

## Results

| Model | Accuracy | Balanced Accuracy | MAE |
|---|---:|---:|---:|
| MedGemma + AnyJev L0 | 37.00% | 25.55% | 0.7839 |
| Raw MedGemma | 36.26% | 24.28% | 0.8095 |
| GPT-5 | 31.50% | 34.73% | 0.9560 |
| XGBoost | 29.30% | 27.78% | 1.2894 |
| Mini-VGG | 22.71% | 16.67% | 1.1209 |

All metrics above use the same 273 matched drawings and corrected labels.

## Findings

- AnyJev L0 improved accuracy by approximately 0.73 percentage points.
- L0 reduced mean absolute error from 0.8095 to 0.7839.
- L0 changed 13 predictions: 10 improved and 3 worsened.
- MedGemma heavily favored score 4, even after calibration.
- GPT-5 achieved higher balanced accuracy than MedGemma.

## Experimental Verification

- All 473 calibration and evaluation images were confirmed to come
  from the authorized NHATS test directory.
- Calibration and evaluation share zero patients.
- All 273 MedGemma predictions match their baseline records.
- Saved L0 probabilities were independently reproduced.
- Real GPU inference reproduced the saved probabilities exactly
  for one image.
- The complete evaluation workflow passed a real-GPU smoke test.
- The full automated test suite passed 131 tests, with 4 skipped
  due to optional dependencies or environment requirements.

## Limitations

- Results are exploratory, not a fully untouched test benchmark.
- Prompt development involved earlier NHATS test examples.
- The matched subset may not represent the entire NHATS population.
- Existing baseline models may use different image resolutions and preprocessing.
- The small improvement from L0 has not been established as statistically significant.
- The multimodal adapter does not implement every feature of AnyJev's
  original text-based Decider.

## Reproducing the Evaluation

Use `examples/evaluate_medgemma_nhats.py` with authorized NHATS data.

Required arguments:
- `--model`: local MedGemma model directory
- `--calibration`: CSV with patient_id and image_path
- `--evaluation`: CSV with patient_id, image_path, true_score
- `--output-dir`: private directory for predictions and prior

The reported experiment used 200 patient-disjoint calibration
drawings, 273 evaluation drawings, Prompt C, and L0 strength 0.75.

The script saves calibration_prior.npy and medgemma_predictions.csv.
Prediction CSVs contain patient identifiers and must remain private.

Requires compatible NumPy, PyTorch, Transformers, and Pillow packages.
Use a CUDA-compatible environment for GPU inference.

## Testing

Run: python -m pytest tests/ -q -rs

Last full-suite result: 131 passed, 4 skipped.
A final test run is required after all changes.

## Data and Privacy

Do not commit NHATS patient images, patient identifiers, prediction CSVs
containing patient IDs, or restricted dataset files to a public repository.
Keep model weights and research data in approved storage.

## Example Usage

The following example requires a locally available MedGemma model
and authorized clock drawing images.

```python
from anyjev.backends.medgemma import MedGemmaBackend
from anyjev.medgemma_decider import MedGemmaDecider

backend = MedGemmaBackend("/path/to/medgemma-4b-it")

prompt = """Evaluate this Clock Drawing Test using NHATS criteria.
0 = Not recognizable as a clock
1 = Severely distorted depiction
2 = Moderately distorted depiction
3 = Mildly distorted depiction
4 = Reasonably accurate depiction
5 = Accurate depiction of a clock
Respond with exactly one digit from 0 to 5."""

decider = MedGemmaDecider(backend, prompt, prior_strength=0.75)

calibration_images = ["/path/to/calibration_clock_1.jpg"]
evaluation_images = ["/path/to/evaluation_clock.jpg"]

decider.calibrate(calibration_images)
decider.save_prior("calibration_prior.npy")

results = decider.predict(evaluation_images)

print("Raw:", results["raw_predictions"])
print("L0:", results["l0_predictions"])
```
