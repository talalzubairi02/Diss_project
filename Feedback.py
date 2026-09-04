import argparse
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marking import predict_score
from Data import load_all_expert_feedback

SCORE_BAND_DESCRIPTIONS = {
    1: "very weak", 2: "weak", 3: "developing",
    4: "adequate", 5: "strong", 6: "very strong",
}

LOCAL_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"  
_model_cache = {}  


def _get_local_model():
    if "model" not in _model_cache:
        print(f"Loading local model {LOCAL_MODEL_NAME} (first call only, may take a minute)...")
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_NAME,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
        )
        _model_cache["tokenizer"] = tokenizer
        _model_cache["model"] = model
        _model_cache["device"] = device
    return _model_cache["model"], _model_cache["tokenizer"], _model_cache["device"]


def call_llm(prompt: str, max_tokens: int = 400) -> str:
    model, tokenizer, device = _get_local_model()

    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to(device)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def retrieve_exemplars(meta_data_root: str, essays_root: str, k: int = 2, seed: int = None) -> list:
    df = load_all_expert_feedback(meta_data_root, essays_root)
    sample = df.sample(n=min(k, len(df)), random_state=seed)
    return list(zip(sample["draft1_text"], sample["expert_feedback"]))

# Condition 1: grounded feedback 

def grounded_feedback(essay_text: str, checkpoint_dir: str, meta_data_root: str, essays_root: str) -> str:
    predicted_score = predict_score(essay_text, checkpoint_dir)
    band = SCORE_BAND_DESCRIPTIONS[predicted_score]
    exemplars = retrieve_exemplars(meta_data_root, essays_root, k=2)

    exemplar_block = ""
    for i, (ex_essay, ex_feedback) in enumerate(exemplars, 1):
        exemplar_block += f"\n--- Example {i} ---\nEssay excerpt: {ex_essay[:400]}...\nExpert feedback given: {ex_feedback}\n"

    prompt = f"""You are an expert writing instructor giving formative feedback on a student's persuasive essay.

An automated marking model has scored this essay as {predicted_score}/6 ({band}).

Here are examples of the style and specificity of feedback an expert writing instructor gives on similar essays:
{exemplar_block}

Now write feedback for this essay, in a comparable style and level of specificity to the examples above:

{essay_text}

Requirements:
- Reference specific content from the essay (a particular claim, piece of evidence, or argument move), not generic praise.
- Keep in mind the essay's overall score band ({band}) when calibrating how much and what kind of feedback to give.
- Keep the feedback to 4-6 sentences.
"""
    return call_llm(prompt)

# Condition 2: raw LLM baseline


def raw_llm_feedback(essay_text: str) -> str:
    prompt = f"""You are giving formative feedback to a student on their persuasive essay.

Essay text:
{essay_text}

Write feedback for the student. Be specific and actionable. Keep it to 4-6 sentences.
"""
    return call_llm(prompt)


# Condition 3: template baseline 

TEMPLATE_COMMENTS = {
    1: "This essay needs significant development. Make sure you clearly state your position and support it with at least one piece of evidence.",
    2: "Your position is present but underdeveloped. Add more evidence and explain how it supports your claim.",
    3: "Your argument is developing. Consider addressing a counterargument to strengthen your position.",
    4: "Your essay makes a clear case. Consider adding a rebuttal to opposing views to make your argument more persuasive.",
    5: "This is a strong essay. Review your evidence for the most compelling examples and ensure your conclusion reinforces your strongest point.",
    6: "This is a very strong essay. Focus on refining word choice and sentence variety for even greater persuasive impact.",
}


def template_feedback(essay_text: str, checkpoint_dir: str) -> str:
    predicted_score = predict_score(essay_text, checkpoint_dir)
    return f"[Score: {predicted_score}/6] " + TEMPLATE_COMMENTS[predicted_score]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--essay", required=True, help="Path to a plain-text essay file")
    parser.add_argument("--mode", choices=["grounded", "raw", "template"], default="grounded")
    parser.add_argument("--checkpoint", default="out/marking_model")
    parser.add_argument(
        "--meta_data_root",
        default="ArgRewrite-main/ArgRewrite-main/dataset/meta-data",
        help="Path to ArgRewrite meta-data folder (needed for grounded mode)",
    )
    parser.add_argument(
        "--essays_root",
        default="ArgRewrite-main/ArgRewrite-main/dataset/essays",
        help="Path to ArgRewrite essays folder (needed for grounded mode)",
    )
    args = parser.parse_args()

    with open(args.essay, "r", encoding="utf-8") as f:
        essay_text = f.read()

    if args.mode == "grounded":
        print(grounded_feedback(essay_text, args.checkpoint, args.meta_data_root, args.essays_root))
    elif args.mode == "raw":
        print(raw_llm_feedback(essay_text))
    else:
        print(template_feedback(essay_text, args.checkpoint))