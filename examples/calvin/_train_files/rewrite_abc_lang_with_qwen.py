"""Rewrite CALVIN ABC's LeRobot ``meta/tasks.jsonl`` into terse imperative
sentences using a *local* Qwen model.

The dataloader (``LeRobotSingleDataset._get_tasks`` /  ``get_language``) reads
``meta/tasks.jsonl`` and looks up ``task_index -> task`` on every step, so a
straight rewrite of this file is the cleanest way to "replace" ABC's language
without any dataloader changes.

The LLM prompt is deliberately scrubbed of any reference to the CALVIN D split.
We only feed the original ABC instruction and a generic *style description*
("short imperative, < 8 words, plain verb, single clause"). No D annotations
are ever placed in the model context. The objective is purely a stylistic
shortening / standardisation of ABC's own instructions, not a copy of D.

Usage::

    python examples/calvin/train_files/rewrite_abc_lang_with_qwen.py \\
        --abc-root /path/to/calvin_task_ABC_D_hf \\
        --model-path /inspire/hdd/global_public/public_models/Qwen/Qwen2.5-7B-Instruct \\
        --backend vllm

A backup of the original file is written to ``meta/tasks.jsonl.orig.bak``
before any overwrite. Use ``--dry-run`` to inspect rewrites without touching
the dataset.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Prompt — does NOT reference CALVIN D or its annotations.
# Style spec + generic, non-D examples.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a strict rephraser. You convert one robot manipulation instruction into\
 a shorter, simpler form.

HARD RULES — never violate any of them:

(R1) Output ONE line only. No commentary, no quotes, no explanation, no labels.
(R2) NOUN PRESERVATION. The output MUST mention the SAME OBJECT as the input.
     - If input says "switch", output says "switch" (NOT "block", "lever").
     - If input says "drawer", output says "drawer" (NOT "shelf", "box").
     - If input says "lightbulb" / "led" / "lamp", keep that exact word.
     - If input says "slider" or "door" or "sliding cabinet", keep one of those.
     - If input says "block", you may keep "block" and the colour.
     - Do NOT invent a colour that is absent from the input.
(R3) VERB FAITHFULNESS. Preserve the action verb or a clear synonym:
     "push X downwards" -> "push X down", NOT "place X".
     "grasp X then open it" -> "open X", NOT "place X".
     "turn it right"        -> "rotate X right" or "turn X right".
     "lift X"               -> "lift X" or "pick up X".
(R4) Preserve every modifier present in the input: colour ("red", "blue",\
 "pink", "green", "yellow"), direction ("left", "right", "down", "up"),\
 location ("in the drawer", "on the table", "on the shelf", "in the slider").
(R5) Length: at most 8 words.
(R6) Plain imperative. No politeness ("please", "could you"). No compound\
 clauses ("then", "after", "and then", "while", "if").
(R7) Do not add a trailing period.

Examples (taken from CALVIN ABC's own instructions — public training data):

  IN:  go towards the pink block in the drawer and pick it up
  OUT: pick up the pink block from the drawer

  IN:  push the switch downwards
  OUT: push the switch down

  IN:  grasp the drawer handle, then open it
  OUT: open the drawer

  IN:  toggle the button to turn on the led
  OUT: turn on the led

  IN:  grasp the red block and turn it right
  OUT: rotate the red block right

  IN:  place the grasped object in the sliding cabinet
  OUT: place the object in the sliding cabinet

  IN:  lift the red block lying on the shelf
  OUT: lift the red block from the shelf

  IN:  slide the door to the left
  OUT: slide the door left
"""

USER_TEMPLATE = "Rewrite:\n{src}"


# ---------------------------------------------------------------------------
# Noun preservation check — semantic sanity guard.
# If the rewrite drops every salient noun/colour/direction that was in the
# source, the rewrite is hallucinated and we fall back to the original.
# ---------------------------------------------------------------------------
_KEY_WORDS = [
    # Objects
    "block", "drawer", "slider", "door", "switch", "button", "led", "light",
    "lightbulb", "lamp", "cabinet", "handle", "shelf", "table", "object",
    # Colours
    "red", "blue", "pink", "green", "yellow", "purple",
    # Directions / positions
    "left", "right", "down", "up", "top", "bottom",
]

# Morphological variants that should normalise to a key word.
_DIRECTION_SYNONYMS = {
    "downwards": "down", "downward": "down",
    "upwards": "up", "upward": "up",
    "leftwards": "left", "leftward": "left",
    "rightwards": "right", "rightward": "right",
    "topmost": "top", "uppermost": "top",
    "lights": "light", "lamps": "lamp", "blocks": "block",
    "drawers": "drawer", "doors": "door", "switches": "switch",
    "buttons": "button", "leds": "led", "objects": "object",
}


def _salient_tokens(s: str) -> set[str]:
    toks = re.findall(r"[a-z]+", s.lower())
    norm: set[str] = set()
    for t in toks:
        canon = _DIRECTION_SYNONYMS.get(t, t)
        if canon in _KEY_WORDS:
            norm.add(canon)
    return norm


def preserves_meaning(src: str, rewrite: str) -> bool:
    """True if the rewrite shares at least the SAME key noun/colour/direction
    set with the source (subset relation: rewrite's salient tokens ⊆ source's,
    AND at least one salient token survived).

    This is intentionally strict: any colour/object that appears in the rewrite
    but not in the source counts as a hallucination.
    """
    src_keys = _salient_tokens(src)
    rw_keys = _salient_tokens(rewrite)
    if not rw_keys:
        # Trivial rewrite with no salient tokens — accept if source also had none.
        return not src_keys
    # No new colour/object/direction introduced.
    if not rw_keys.issubset(src_keys):
        return False
    # At least one key noun preserved (avoid empty rewrite passing).
    return len(rw_keys) > 0 if src_keys else True


# ---------------------------------------------------------------------------
# Output post-processing
# ---------------------------------------------------------------------------
_BAD_PREFIXES = re.compile(
    r"^\s*(OUT\s*:\s*|Rewrite\s*:\s*|Answer\s*:\s*|>|\*+)", re.IGNORECASE
)
_QUOTE_CHARS = "\"“”‘’`"
# Chatter often produced by Qwen before the actual answer.
_INTERJECTIONS = re.compile(
    r"^\s*(sure|okay|ok|certainly|of course|alright|here\s+is(\s+the)?\s+rewrite\s*[:,]?)\s*[!,.\-:]?\s*",
    re.IGNORECASE,
)
# Bare chat role markers that Qwen occasionally bleeds through when the chat
# template terminator is mis-placed. Lines like "user", "Assistant:", "Human:".
_ROLE_NAMES = r"user|assistant|system|human"
_ROLE_MARKER = re.compile(rf"^\s*({_ROLE_NAMES})\s*[:.]?\s*$", re.IGNORECASE)
# Inline form: "Assistant: turn off the led" — strip the role prefix only.
_ROLE_INLINE = re.compile(rf"^\s*({_ROLE_NAMES})\s*[:.\-]\s+", re.IGNORECASE)


def clean_output(text: str, src: str) -> str:
    """Strip the LLM's extra padding and enforce the style envelope.

    Returns the source string if the model produced something obviously off
    (>15 words, empty, repeating the input verbatim with politeness padding,
    or only role markers).
    """
    # Collect non-empty lines and discard pure role-marker lines.
    raw_lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    lines = [ln for ln in raw_lines if not _ROLE_MARKER.match(ln)]
    line = lines[0] if lines else ""
    line = _ROLE_INLINE.sub("", line).strip()
    line = _BAD_PREFIXES.sub("", line).strip()
    # Strip leading chatter ("Sure!", "Okay,", "Here is the rewrite:") possibly multiple layers.
    for _ in range(3):
        new_line = _INTERJECTIONS.sub("", line)
        if new_line == line:
            break
        line = new_line.strip()
    line = line.strip(_QUOTE_CHARS).strip()
    if line.endswith("."):
        line = line[:-1].rstrip()
    if not line:
        return src
    if len(line.split()) > 15:
        return src
    return line


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
def run_vllm(model_path: str, prompts: list[tuple[int, str]], tensor_parallel_size: int = 2) -> dict[int, str]:
    from vllm import LLM, SamplingParams  # type: ignore

    print(f"[vllm] Loading {model_path} with tensor_parallel_size={tensor_parallel_size}...")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        tensor_parallel_size=tensor_parallel_size,
    )
    # Use Qwen's actual chat template stop strings. Without these vllm keeps
    # generating past <|im_end|> and emits the next turn's "user\n..." prefix,
    # which then becomes the visible model output.
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=64,
        stop=["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
    )

    # Build chat-formatted prompts via the model's tokenizer.
    tok = llm.get_tokenizer()
    chat_prompts: list[str] = []
    for _, src in prompts:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(src=src)},
        ]
        chat_prompts.append(
            tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        )

    print(f"[vllm] Generating {len(chat_prompts)} paraphrases...")
    outs = llm.generate(chat_prompts, sampling)
    result: dict[int, str] = {}
    for (idx, src), out in zip(prompts, outs):
        text = out.outputs[0].text if out.outputs else ""
        result[idx] = clean_output(text, src)
    return result


def run_transformers(model_path: str, prompts: list[tuple[int, str]], batch_size: int = 8) -> dict[int, str]:
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    from tqdm import tqdm  # type: ignore

    print(f"[transformers] Loading {model_path}...")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Add Qwen's <|im_end|> as an additional stop id so generation halts at end
    # of the assistant turn rather than continuing into the next "user" turn.
    extra_eos_ids: list[int] = []
    for special in ("<|im_end|>", "<|endoftext|>"):
        tid = tok.convert_tokens_to_ids(special)
        if isinstance(tid, int) and tid >= 0:
            extra_eos_ids.append(tid)
    eos_token_id = list({tok.eos_token_id, *extra_eos_ids})

    def build(src: str) -> str:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(src=src)},
        ]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    result: dict[int, str] = {}
    for start in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch = prompts[start : start + batch_size]
        texts = [build(src) for _, src in batch]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=64,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tok.pad_token_id,
                eos_token_id=eos_token_id,
            )
        for (idx, src), input_ids, gen_ids in zip(batch, enc["input_ids"], out):
            new_tokens = gen_ids[input_ids.shape[0] :]
            text = tok.decode(new_tokens, skip_special_tokens=True)
            result[idx] = clean_output(text, src)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--abc-root", required=True, type=Path,
                        help="Path to calvin_task_ABC_D_hf (the LeRobot dataset).")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-72B-Instruct",
        help="Local Qwen Instruct model path. Default is the 72B variant — runs "
             "comfortably on 2x H200 in bf16 (~144 GB) via vllm tensor parallel. "
             "For an even stronger MoE option try "
             "/inspire/hdd/global_public/public_models/Qwen/Qwen3-235B-A22B-Instruct-2507.",
    )
    parser.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--tensor-parallel-size", type=int, default=2,
                        help="vllm tensor-parallel world size. 2 fits Qwen2.5-72B "
                             "in bf16 on 2x H200 with headroom; 8 lets you run "
                             "Qwen3-235B-A22B-Instruct-2507.")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for the transformers backend.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print rewrites, do not write the output file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "abc_tasks_d_style.jsonl",
        help="Output path for the rewritten tasks.jsonl. "
             "Defaults to <starVLA>/examples/calvin/train_files/abc_tasks_d_style.jsonl. "
             "The script never writes inside --abc-root unless you explicitly "
             "point --output there.",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Only rewrite the first N tasks (debugging).")
    args = parser.parse_args()

    abc_root: Path = args.abc_root
    tasks_path = abc_root / "meta" / "tasks.jsonl"
    if not tasks_path.is_file():
        raise FileNotFoundError(f"Missing {tasks_path}")

    # 1) Read original tasks
    with open(tasks_path) as f:
        entries = [json.loads(line) for line in f]
    print(f"Loaded {len(entries)} tasks from {tasks_path}")

    if args.limit:
        entries = entries[: args.limit]

    # 2) Build (task_index, original_text) prompt list, dedup by text
    text_to_indices: dict[str, list[int]] = {}
    for e in entries:
        text_to_indices.setdefault(e["task"], []).append(int(e["task_index"]))
    print(f"Unique task texts: {len(text_to_indices)}")

    prompts: list[tuple[int, str]] = [
        (idx_list[0], text) for text, idx_list in text_to_indices.items()
    ]

    # 3) Generate
    if args.backend == "vllm":
        try:
            unique_rewrites = run_vllm(args.model_path, prompts, args.tensor_parallel_size)
        except Exception as e:
            print(f"[warn] vllm failed ({e}); falling back to transformers.")
            unique_rewrites = run_transformers(args.model_path, prompts, args.batch_size)
    else:
        unique_rewrites = run_transformers(args.model_path, prompts, args.batch_size)

    # 3b) Semantic sanity guard: reject hallucinated rewrites that introduce
    #     colours/objects/directions not present in the source, or that drop
    #     every salient noun from a non-empty source.
    n_rejected = 0
    for idx, src in prompts:
        rw = unique_rewrites[idx]
        if not preserves_meaning(src, rw):
            unique_rewrites[idx] = src
            n_rejected += 1
    if n_rejected:
        print(f"[guard] {n_rejected}/{len(prompts)} rewrites failed noun-preservation "
              "check; kept the original source instruction for those.")

    # 4) Project back to all task_index rows
    rewrites: dict[int, str] = {}
    for (rep_idx, src), text in zip(prompts, [unique_rewrites[i] for i, _ in prompts]):
        for idx in text_to_indices[src]:
            rewrites[idx] = text

    # 5) Print a sample for human review
    print("\n=== Sample rewrites (first 20 unique) ===")
    for src, idx_list in list(text_to_indices.items())[:20]:
        print(f"  [{idx_list[0]:4d}] {src!r}\n         -> {rewrites[idx_list[0]]!r}")

    if args.dry_run:
        print("\n--dry-run set, NOT writing output file.")
        return

    # 6) Write the rewritten tasks.jsonl to the user-chosen --output location.
    # By default this is INSIDE the starVLA repo (examples/calvin/train_files/),
    # NOT inside --abc-root, so the shared read-only dataset is never modified.
    out_path: Path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path == tasks_path:
        # User explicitly requested in-place. Back up the original first.
        backup = tasks_path.with_suffix(".jsonl.orig.bak")
        if not backup.exists():
            shutil.copy2(tasks_path, backup)
            print(f"Backed up original → {backup}")
        else:
            print(f"Backup already exists at {backup}; skipping copy.")

    new_entries = []
    for e in entries:
        new_entries.append({"task_index": e["task_index"], "task": rewrites[int(e["task_index"])]})
    with open(out_path, "w") as f:
        for e in new_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"Wrote {len(new_entries)} rewritten tasks → {out_path}")

    if out_path != tasks_path:
        print(
            "\nTo use this file at training time, add the following to your\n"
            "data config YAML (e.g. examples/calvin/train_files/starvla_train_calvin.yaml):\n\n"
            "    datasets:\n"
            "      vla_data:\n"
            f"        tasks_override_path: {out_path}\n"
        )


if __name__ == "__main__":
    main()
