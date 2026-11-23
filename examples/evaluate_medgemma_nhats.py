"""Evaluate MedGemma + AnyJev L0 on patient-disjoint NHATS datasets."""

import argparse
import csv
from pathlib import Path

import numpy as np

from anyjev.backends.medgemma import MedGemmaBackend
from anyjev.medgemma_decider import MedGemmaDecider


PROMPT = """You are evaluating a Clock Drawing Test using NHATS scoring criteria.

Examine the drawing carefully:
- Is there a recognizable clock outline?
- Are the numbers present and arranged in appropriate clock positions?
- Are there recognizable clock hands?
- How severely is the overall clock structure distorted?

Use the following scoring scale:
0 = Not recognizable as a clock
1 = Severely distorted depiction
2 = Moderately distorted depiction
3 = Mildly distorted depiction
4 = Reasonably accurate depiction
5 = Accurate depiction of a clock

Do not assume a drawing is accurate merely because it contains a circle or numbers.
Do not provide explanations.
Respond with exactly one digit from 0 to 5."""


def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cal = load_rows(args.calibration)
    evaluation = load_rows(args.evaluation)

    if not cal or not evaluation:
        raise ValueError("Calibration and evaluation must be nonempty.")

    for name, rows in [("calibration", cal), ("evaluation", evaluation)]:
        required = {"patient_id", "image_path"}
        if not required.issubset(rows[0]):
            raise ValueError(f"{name} CSV missing required columns.")

    if "true_score" not in evaluation[0]:
        raise ValueError("Evaluation CSV must include true_score.")

    cal_patients = {r["patient_id"] for r in cal}
    eval_patients = {r["patient_id"] for r in evaluation}

    if cal_patients & eval_patients:
        raise ValueError("Patient overlap between calibration and evaluation.")

    cal_paths = {r["image_path"] for r in cal}
    eval_paths = {r["image_path"] for r in evaluation}

    if cal_paths & eval_paths:
        raise ValueError("Image overlap between calibration and evaluation.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    backend = MedGemmaBackend(args.model)
    decider = MedGemmaDecider(backend, PROMPT, prior_strength=0.75)

    prior = decider.calibrate([r["image_path"] for r in cal])
    decider.save_prior(args.output_dir / "calibration_prior.npy")
    print("Calibration prior:", repr(prior), flush=True)

    results = decider.predict([r["image_path"] for r in evaluation])

    scores = np.array(
        [float(r["true_score"]) for r in evaluation],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(scores)):
        raise ValueError("Ground-truth scores must be finite.")
    if not np.all((scores >= 0) & (scores <= 5) & (scores == np.floor(scores))):
        raise ValueError("Ground-truth scores must be integers from 0 to 5.")
    truth = scores.astype(int)

    for name, predictions in [
        ("Raw", results["raw_predictions"]),
        ("L0", results["l0_predictions"]),
    ]:
        accuracy = float(np.mean(predictions == truth))
        mae = float(np.mean(np.abs(predictions - truth)))
        print(f"{name}: Accuracy={accuracy:.4f}, MAE={mae:.4f}", flush=True)

    fields = [
        "patient_id", "round", "image_path", "true_score",
        "raw_prediction", "l0_prediction",
    ] + [f"raw_prob_{i}" for i in range(6)] + [
        f"l0_prob_{i}" for i in range(6)
    ]

    output = args.output_dir / "medgemma_predictions.csv"

    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for i, row in enumerate(evaluation):
            record = {
                "patient_id": row["patient_id"],
                "round": row.get("round", ""),
                "image_path": row["image_path"],
                "true_score": int(float(row["true_score"])),
                "raw_prediction": int(results["raw_predictions"][i]),
                "l0_prediction": int(results["l0_predictions"][i]),
            }

            for score in range(6):
                record[f"raw_prob_{score}"] = float(
                    results["raw_probs"][i, score]
                )
                record[f"l0_prob_{score}"] = float(
                    results["l0_probs"][i, score]
                )

            writer.writerow(record)

    print("Predictions saved:", output, flush=True)


if __name__ == "__main__":
    main()
