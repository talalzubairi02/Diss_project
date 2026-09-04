import argparse
import json
import re

import pandas as pd

from Feedback import call_llm

STOPWORDS = set("""
a an the and or but if then so because as until while of at by for with
about against between into through during before after above below to from
up down in out on off over under again further once here there when where
why how all any both each few more most other some such no nor not only own
same so than too very s t can will just don should now is are was were be
been being have has had do does did this that these those i you he she it
we they them his her its our your their what which who whom
""".split())


def content_words(text: str) -> set:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def lexical_grounding_score(essay_text: str, feedback_text: str) -> float:
    essay_words = content_words(essay_text)
    feedback_words = content_words(feedback_text)
    if not feedback_words:
        return 0.0
    overlap = essay_words & feedback_words
    return len(overlap) / len(feedback_words)


JUDGE_PROMPT_TEMPLATE = """You are scoring feedback given to a student on their persuasive essay.

Essay:
{essay_text}

Feedback given to the student:
{feedback_text}

Rate the feedback on three dimensions, each from 1 (very poor) to 5 (excellent):
- specificity: does it reference concrete parts of THIS essay, rather than generic advice that could apply to any essay?
- helpfulness: would acting on this feedback plausibly improve the essay?
- validity: is the feedback factually accurate about what the essay actually contains?

Respond with ONLY a JSON object in this exact format, nothing else:
{{"specificity": <1-5>, "helpfulness": <1-5>, "validity": <1-5>}}
"""


def parse_judge_response(text: str) -> dict:
    match = re.search(r"\{[^{}]*\}", text)
    if not match:
        return {"specificity": None, "helpfulness": None, "validity": None}
    try:
        parsed = json.loads(match.group(0))
        return {
            "specificity": parsed.get("specificity"),
            "helpfulness": parsed.get("helpfulness"),
            "validity": parsed.get("validity"),
        }
    except json.JSONDecodeError:
        return {"specificity": None, "helpfulness": None, "validity": None}


def judge_feedback(essay_text: str, feedback_text: str) -> dict:
    prompt = JUDGE_PROMPT_TEMPLATE.format(essay_text=essay_text, feedback_text=feedback_text)
    response = call_llm(prompt, max_tokens=100)
    return parse_judge_response(response)


def main(args):
    df = pd.read_csv(args.input)
    print(f"Scoring {len(df)} feedback items...")

    results = []
    for i, row in df.iterrows():
        essay_text = row["essay_text"]
        feedback_text = row["feedback_text"]

        grounding = lexical_grounding_score(essay_text, feedback_text)
        word_count = len(feedback_text.split())

        # Two independent judge passes, per the disagreement-reporting design
        judge_1 = judge_feedback(essay_text, feedback_text)
        judge_2 = judge_feedback(essay_text, feedback_text)

        print(f"[{i+1}/{len(df)}] essay_id={row['essay_id']} condition={row['condition']} "
              f"grounding={grounding:.2f} judge1={judge_1} judge2={judge_2}")

        results.append({
            "essay_id": row["essay_id"],
            "condition": row["condition"],
            "true_score": row["true_score"],
            "predicted_score": row["predicted_score"],
            "word_count": word_count,
            "lexical_grounding": grounding,
            "specificity_1": judge_1["specificity"],
            "helpfulness_1": judge_1["helpfulness"],
            "validity_1": judge_1["validity"],
            "specificity_2": judge_2["specificity"],
            "helpfulness_2": judge_2["helpfulness"],
            "validity_2": judge_2["validity"],
        })

    scores_df = pd.DataFrame(results)

    for dim in ["specificity", "helpfulness", "validity"]:
        scores_df[f"{dim}_mean"] = scores_df[[f"{dim}_1", f"{dim}_2"]].mean(axis=1)
        scores_df[f"{dim}_disagreement"] = (scores_df[f"{dim}_1"] - scores_df[f"{dim}_2"]).abs()

    scores_df.to_csv(args.output, index=False)
    print(f"\nSaved per-item scores to {args.output}")

    summary = scores_df.groupby("condition").agg(
        n=("essay_id", "count"),
        avg_word_count=("word_count", "mean"),
        avg_lexical_grounding=("lexical_grounding", "mean"),
        avg_specificity=("specificity_mean", "mean"),
        avg_helpfulness=("helpfulness_mean", "mean"),
        avg_validity=("validity_mean", "mean"),
        avg_specificity_disagreement=("specificity_disagreement", "mean"),
        avg_helpfulness_disagreement=("helpfulness_disagreement", "mean"),
        avg_validity_disagreement=("validity_disagreement", "mean"),
    ).round(2)

    print("\n=== Summary by condition (report-ready) ===")
    print(summary.to_string())
    summary.to_csv(args.output.replace(".csv", "_summary.csv"))
    print(f"\nSaved summary table to {args.output.replace('.csv', '_summary.csv')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV from generate_feedback_batch.py")
    parser.add_argument("--output", default="feedback_scores.csv")
    args = parser.parse_args()
    main(args)