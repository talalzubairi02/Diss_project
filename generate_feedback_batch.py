import argparse

import pandas as pd

from Data import load_persuade
from Marking import predict_score
from Feedback import grounded_feedback, raw_llm_feedback, template_feedback


def sample_essays(csv_path: str, n_per_score: int, seed: int = 42) -> pd.DataFrame:
    df = load_persuade(csv_path)
    df = df.dropna(subset=["holistic_essay_score", "full_text"])
    sampled = (
        df.groupby("holistic_essay_score", group_keys=False)
        .apply(lambda g: g.sample(n=min(n_per_score, len(g)), random_state=seed))
    )
    return sampled.reset_index(drop=True)


def main(args):
    essays = sample_essays(args.csv, args.n_per_score)
    print(f"Generating feedback for {len(essays)} essays across all score bands...")

    rows = []
    for i, row in essays.iterrows():
        essay_id = row["essay_id"]
        essay_text = row["full_text"]
        true_score = row["holistic_essay_score"]
        pred_score = predict_score(essay_text, args.checkpoint)

        print(f"[{i+1}/{len(essays)}] essay_id={essay_id} true_score={true_score} pred_score={pred_score}")

        conditions = {
            "grounded": lambda: grounded_feedback(essay_text, args.checkpoint, args.meta_data_root, args.essays_root),
            "raw": lambda: raw_llm_feedback(essay_text),
            "template": lambda: template_feedback(essay_text, args.checkpoint),
        }

        for condition_name, fn in conditions.items():
            feedback_text = fn()
            rows.append({
                "essay_id": essay_id,
                "true_score": true_score,
                "predicted_score": pred_score,
                "condition": condition_name,
                "essay_text": essay_text,
                "feedback_text": feedback_text,
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, index=False)
    print(f"\nSaved {len(out_df)} feedback rows ({len(essays)} essays x 3 conditions) to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to PERSUADE CSV")
    parser.add_argument("--checkpoint", default="out/marking_model")
    parser.add_argument("--meta_data_root", default="argrewrite_data/ArgRewrite-main/dataset/meta-data")
    parser.add_argument("--essays_root", default="argrewrite_data/ArgRewrite-main/dataset/essays")
    parser.add_argument("--n_per_score", type=int, default=3, help="How many essays to sample per score band (1-6)")
    parser.add_argument("--output", default="feedback_batch.csv")
    args = parser.parse_args()
    main(args)