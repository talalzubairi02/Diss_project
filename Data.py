import os
import re

import pandas as pd

# PERSUADE 2.0

def load_persuade(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"essay_id_comp": "essay_id"})
    required = ["essay_id", "full_text", "holistic_essay_score", "prompt_name"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Expected columns {missing} not found. Got: {list(df.columns)}")
    return df


# ArgRewrite V.2

_ESSAY_FNAME_RE = re.compile(r"draft(\d)_2018argrewrite_(\d+)\.txt")


def list_argrewrite_ids(essays_root: str) -> list:
    ids = []
    for fname in os.listdir(os.path.join(essays_root, "Draft1")):
        m = _ESSAY_FNAME_RE.match(fname)
        if m:
            ids.append(int(m.group(2)))
    return sorted(ids)


def load_argrewrite_essay(essay_id: int, draft: int, essays_root: str) -> str:
    path = os.path.join(essays_root, f"Draft{draft}", f"draft{draft}_2018argrewrite_{essay_id}.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def find_argrewrite_interface(essay_id: int, annotations_root: str, transition: str = "12") -> str:
    for interface in ["A", "B", "C", "D"]:
        path = os.path.join(
            annotations_root, interface, transition,
            f"Annotation_2018argrewrite_{essay_id}_NEW.txt.xlsx",
        )
        if os.path.exists(path):
            return interface
    return None


def load_argrewrite_annotation(essay_id: int, annotations_root: str, transition: str = "12") -> dict:
    interface = find_argrewrite_interface(essay_id, annotations_root, transition)
    if interface is None:
        return None
    path = os.path.join(
        annotations_root, interface, transition,
        f"Annotation_2018argrewrite_{essay_id}_NEW.txt.xlsx",
    )
    xl = pd.ExcelFile(path)
    return {
        "interface": interface,
        "old": xl.parse("Old Draft"),
        "new": xl.parse("New Draft"),
    }


def load_argrewrite_expert_feedback(essay_id: int, meta_data_root: str) -> str:
    try:
        import docx  # python-docx
    except ImportError:
        raise ImportError("pip install python-docx")
    path = os.path.join(meta_data_root, "ExpertFeedback", f"feedback_{essay_id}.docx")
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def load_all_expert_feedback(meta_data_root: str, essays_root: str) -> pd.DataFrame:
    rows = []
    for essay_id in list_argrewrite_ids(essays_root):
        fb_path = os.path.join(meta_data_root, "ExpertFeedback", f"feedback_{essay_id}.docx")
        if not os.path.exists(fb_path):
            continue
        draft1_text = load_argrewrite_essay(essay_id, 1, essays_root)
        feedback_text = load_argrewrite_expert_feedback(essay_id, meta_data_root)
        rows.append({"essay_id": essay_id, "draft1_text": draft1_text, "expert_feedback": feedback_text})
    return pd.DataFrame(rows)


def load_argrewrite_scores(scores_path: str) -> pd.DataFrame:
    df = pd.read_excel(scores_path)
    df["writer"] = df["writer"].ffill()
    return df


if __name__ == "__main__":
    persuade_df = load_persuade("persuade_2.0_human_scores_demo_id_github/persuade_2.0_human_scores_demo_id_github.csv")
    print("PERSUADE loaded:", persuade_df.shape)
    print(persuade_df.columns.tolist())

    essays_root = "ArgRewrite-main/ArgRewrite-main/dataset/essays"
    annotations_root = "ArgRewrite-main/ArgRewrite-main/dataset/annotations"
    meta_data_root = "ArgRewrite-main/ArgRewrite-main/dataset/meta-data"

    ids = list_argrewrite_ids(essays_root)
    print(f"{len(ids)} ArgRewrite ids found:", ids[:5])

    ann = load_argrewrite_annotation(ids[0], annotations_root)
    print("First annotation interface:", ann["interface"])
    print(ann["old"].head())

    fb = load_argrewrite_expert_feedback(ids[0], meta_data_root)
    print("First expert feedback:", fb[:200])