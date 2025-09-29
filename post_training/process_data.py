# preprocess_jsonl.py
import json
import random
from pathlib import Path
from datasets import Dataset
from typing import List, Dict
import argparse

DEFAULT_PROMPT_TMPL = (
    "{instruction}\n"
    "Grid shape: {grid_shape}\n"
    "No-fly zones: {noflyzone}\n"
    "{abstract_map_section}"
    "Produce the low-level action plan (one-line) as the output.\n"
    "Output:"
)

def build_prompt(example: Dict, include_abstract: bool):
    abstract_text = ""
    if include_abstract and example.get("abstract_map"):
        abstract_text = f"Abstract map: {example['abstract_map']}\n"
    return DEFAULT_PROMPT_TMPL.format(
        instruction=example.get("instruction",""),
        planner=example.get("planner",""),
        grid_shape=example.get("grid_shape",[]),
        start=example.get("start",[]),
        goal=example.get("goal",[]),
        altitude=example.get("altitude",""),
        noflyzone=example.get("noflyzone",""),
        abstract_map_section=abstract_text
    )

def build_dataset_from_jsonl(input_path: str, output_path: str, p_include_abstract: float = 0.6, seed: int = 42):
    random.seed(seed)
    lines = Path(input_path).read_text().strip().splitlines()
    prompts = []
    responses = []
    full_records = []
    for ln in lines:
        j = json.loads(ln)
        # If user included "noflyzone" in JSON occasionally as field name
        nofly = j.get("noflyzone", j.get("noflyzone",""))
        # choose whether to include abstract_map
        include_abs = random.random() < p_include_abstract
        prompt = build_prompt({**j, "noflyzone": nofly}, include_abs)
        output = j.get("output","")
        # Optionally you can append metadata
        prompts.append(prompt)
        responses.append(output)
        full_records.append({
            "prompt": prompt,
            "response": output,
            **j
        })

    ds = Dataset.from_dict({"prompt": prompts, "response": responses})
    ds = ds.train_test_split(test_size=0.02, seed=seed)  # small val split
    # save intermediate file(s)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    ds["train"].to_json(str(Path(output_path) / "train.jsonl"), lines=True)
    ds["test"].to_json(str(Path(output_path) / "valid.jsonl"), lines=True)
    print(f"Saved train/valid to {output_path}")
    return ds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="input jsonl file (one record per line)")
    parser.add_argument("--outdir", type=str, required=True, help="output directory")
    parser.add_argument("--p_abstract", type=float, default=0.6, help="probability to include abstract_map in prompt")
    args = parser.parse_args()
    build_dataset_from_jsonl(args.input, args.outdir, args.p_abstract)

