#!/usr/bin/env python3
"""Phase 1a — inspect Lance weight topology.

Dumps the safetensors key list, classifies keys by component, and resolves the
five open architectural questions that affect the MLX implementation:

    Q1: Are MaPE Δ_m offsets LEARNED (in checkpoint) or HARD-CODED (in config)?
    Q2: Is the LM head UNTIED from input embeddings? (Expected: yes)
    Q3: What is the flow head structure? (linear / MLP / DiT block)
    Q4: Are attention QKV projections SHARED across UND/GEN or DUPLICATED?
    Q5: How many position groups does MaPE actually use? (paper implies 4)

Run after Phase 0 downloads:
    huggingface-cli download bytedance-research/Lance --local-dir ~/models/Lance

Then:
    uv run python scripts/01_inspect_keys.py \\
        --checkpoint ~/models/Lance/Lance_3B/model.safetensors

Output:
    notes/lance_keys_full.txt          full key list (key, shape, dtype, numel)
    notes/lance_keys_summary.md        component breakdown + Q1-Q5 answers
    notes/lance_architecture.md        derived architectural facts
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from safetensors import safe_open


# Component classification — prefix → label. Order matters (first match wins).
# Adjust patterns once empirical key list reveals upstream naming.
COMPONENT_PATTERNS = [
    (re.compile(r"^model\.embed_tokens\.|^embed_tokens\."), "embeddings"),
    (re.compile(r"^lm_head\.|^model\.lm_head\."), "lm_head"),
    # Per-expert: look for explicit _und / _gen suffixes
    (re.compile(r".*\.ffn_und\.|.*\.mlp_und\.|.*_und\.gate_proj|.*_und\.up_proj|.*_und\.down_proj"), "ffn_und"),
    (re.compile(r".*\.ffn_gen\.|.*\.mlp_gen\.|.*_gen\.gate_proj|.*_gen\.up_proj|.*_gen\.down_proj"), "ffn_gen"),
    (re.compile(r".*\.qk_norm_und\.|.*q_norm_und|.*k_norm_und"), "qk_norm_und"),
    (re.compile(r".*\.qk_norm_gen\.|.*q_norm_gen|.*k_norm_gen"), "qk_norm_gen"),
    (re.compile(r".*\.o_proj_und\.|.*self_attn\.o_proj_und"), "o_proj_und"),
    (re.compile(r".*\.o_proj_gen\.|.*self_attn\.o_proj_gen"), "o_proj_gen"),
    # Shared attention (QKV, possibly o_proj if not split)
    (re.compile(r".*\.self_attn\.q_proj\.|.*\.self_attn\.k_proj\.|.*\.self_attn\.v_proj\.|.*\.self_attn\.o_proj\."), "attn_shared"),
    (re.compile(r".*\.self_attn\."), "attn_other"),
    # VAE — may be bundled or separate file
    (re.compile(r"^vae\.|^model\.vae\."), "vae"),
    # ViT
    (re.compile(r"^visual\.|^vit\.|^model\.visual\."), "vit"),
    # Flow head
    (re.compile(r"^flow_head\.|^velocity_head\.|^gen_head\."), "flow_head"),
    # MaPE
    (re.compile(r"^mape\.|^model\.mape_offsets|.*delta_m"), "mape"),
    # Connectors
    (re.compile(r"^connector\.|.*\.connector\."), "connector"),
    # Norms (catch-all after per-expert)
    (re.compile(r".*\.input_layernorm\.|.*\.post_attention_layernorm\.|.*\.norm\."), "norm"),
    (re.compile(r"^model\.norm\.|^final_norm\."), "final_norm"),
]


def classify(key: str) -> str:
    for pattern, label in COMPONENT_PATTERNS:
        if pattern.match(key):
            return label
    return "unclassified"


def analyze_q1_mape(by_component: dict[str, list[str]], shapes: dict[str, list[int]]) -> dict:
    """Q1: Are MaPE Δ_m offsets learned or hard-coded?"""
    mape_keys = by_component.get("mape", [])
    if not mape_keys:
        return {"q1_verdict": "HARD_CODED", "evidence": "No mape.* keys in checkpoint",
                "next_step": "Read llm_config.json for the Δ_m constants"}
    return {"q1_verdict": "LEARNED", "evidence": f"Found {len(mape_keys)} mape keys: {mape_keys[:5]}",
            "shapes": {k: shapes[k] for k in mape_keys[:5]},
            "next_step": "Load via convert.py; pass to MaPEOffsets(learned=True)"}


def analyze_q2_lm_head(shapes: dict[str, list[int]]) -> dict:
    """Q2: Is lm_head untied from embed_tokens?"""
    embed = next((s for k, s in shapes.items() if "embed_tokens" in k), None)
    lm_head = next((s for k, s in shapes.items() if "lm_head" in k), None)
    if embed is None:
        return {"q2_verdict": "UNKNOWN", "evidence": "no embed_tokens key found"}
    if lm_head is None:
        return {"q2_verdict": "TIED", "evidence": "no separate lm_head key — tied to embed_tokens"}
    return {"q2_verdict": "UNTIED", "evidence": f"separate lm_head {lm_head} vs embed {embed}",
            "next_step": "Load lm_head as independent parameter in LanceModel"}


def analyze_q3_flow_head(by_component: dict[str, list[str]], shapes: dict[str, list[int]]) -> dict:
    """Q3: What's the flow head structure?"""
    fh_keys = by_component.get("flow_head", [])
    if not fh_keys:
        return {"q3_verdict": "UNKNOWN",
                "next_step": "Run script with --verbose to grep all keys for 'velocity', 'flow', 'gen_head'"}
    fh_shapes = {k: shapes[k] for k in fh_keys}
    n_keys = len(fh_keys)
    if n_keys <= 2:
        return {"q3_verdict": "LINEAR_PROJECTION", "shapes": fh_shapes, "n_keys": n_keys}
    if n_keys <= 6:
        return {"q3_verdict": "MLP", "shapes": fh_shapes, "n_keys": n_keys}
    return {"q3_verdict": "DIT_BLOCK_OR_LARGER", "shapes": fh_shapes, "n_keys": n_keys,
            "next_step": "Inspect key names for AdaLN / time_embed / cross_attn signatures"}


def analyze_q4_attn(by_component: dict[str, list[str]]) -> dict:
    """Q4: Are attention QKV projections shared or duplicated?"""
    shared_attn = by_component.get("attn_shared", [])
    o_und = by_component.get("o_proj_und", [])
    o_gen = by_component.get("o_proj_gen", [])
    if shared_attn and not o_und and not o_gen:
        return {"q4_verdict": "FULLY_SHARED",
                "evidence": f"{len(shared_attn)} shared attn keys, no per-expert o_proj"}
    if o_und and o_gen:
        return {"q4_verdict": "QKV_SHARED_OPROJ_SPLIT",
                "evidence": f"{len(o_und)} o_proj_und + {len(o_gen)} o_proj_gen"}
    return {"q4_verdict": "AMBIGUOUS",
            "shared_count": len(shared_attn), "und_count": len(o_und), "gen_count": len(o_gen),
            "next_step": "Manually inspect a few attention layer keys"}


def analyze_q5_position_groups(by_component: dict[str, list[str]], shapes: dict[str, list[int]]) -> dict:
    """Q5: How many MaPE position groups?"""
    mape_keys = by_component.get("mape", [])
    for k in mape_keys:
        if "delta_m" in k or "offset" in k:
            return {"q5_verdict": "LEARNED",
                    "num_groups": shapes[k][0] if shapes[k] else "unknown",
                    "evidence": f"{k} has shape {shapes[k]}"}
    return {"q5_verdict": "CONSULT_CONFIG",
            "next_step": "Read llm_config.json for 'mape_num_groups' or 'num_modality_groups'"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to model.safetensors (e.g. ~/models/Lance/Lance_3B/model.safetensors)")
    parser.add_argument("--notes-dir", type=Path, default=Path("notes"))
    parser.add_argument("--name", default="lance")
    parser.add_argument("--verbose", action="store_true", help="Dump unclassified keys")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"ERROR: {args.checkpoint} not found", file=sys.stderr)
        return 1

    args.notes_dir.mkdir(parents=True, exist_ok=True)
    full_txt = args.notes_dir / f"{args.name}_keys_full.txt"
    summary_md = args.notes_dir / f"{args.name}_keys_summary.md"
    arch_md = args.notes_dir / f"{args.name}_architecture.md"

    by_component: dict[str, list[str]] = defaultdict(list)
    shapes: dict[str, list[int]] = {}
    dtypes: Counter[str] = Counter()
    total_params = 0

    with safe_open(str(args.checkpoint), framework="numpy") as f, full_txt.open("w") as out:
        keys = list(f.keys())
        out.write(f"# {args.checkpoint}\n# {len(keys)} tensors\n\n")
        for key in keys:
            slice_ = f.get_slice(key)
            shape = slice_.get_shape()
            dtype = str(slice_.get_dtype())
            numel = 1
            for d in shape:
                numel *= d
            total_params += numel
            dtypes[dtype] += 1
            shapes[key] = shape
            by_component[classify(key)].append(key)
            out.write(f"{key}\t{shape}\t{dtype}\t{numel:,}\n")

    # Resolve the five open questions
    q1 = analyze_q1_mape(by_component, shapes)
    q2 = analyze_q2_lm_head(shapes)
    q3 = analyze_q3_flow_head(by_component, shapes)
    q4 = analyze_q4_attn(by_component)
    q5 = analyze_q5_position_groups(by_component, shapes)

    # Component breakdown summary
    with summary_md.open("w") as out:
        out.write(f"# {args.name} — key topology summary\n\n")
        out.write(f"- Checkpoint: `{args.checkpoint.name}`\n")
        out.write(f"- Total tensors: **{len(shapes)}**\n")
        out.write(f"- Total parameters: **{total_params / 1e9:.2f} B**\n")
        out.write(f"- Dtypes: {', '.join(f'{d}={n}' for d, n in dtypes.most_common())}\n\n")

        out.write("## Component breakdown\n\n")
        for component in sorted(by_component, key=lambda c: -len(by_component[c])):
            keys_in = by_component[component]
            params_in = sum(
                int(__import__("functools").reduce(lambda a, b: a * b, shapes[k], 1))
                for k in keys_in
            )
            out.write(f"### `{component}` — {len(keys_in)} tensors, ~{params_in / 1e9:.2f} B params\n\n")
            for k in keys_in[:5]:
                out.write(f"- `{k}` {shapes[k]}\n")
            if len(keys_in) > 5:
                out.write(f"- ... and {len(keys_in) - 5} more\n")
            out.write("\n")

        if args.verbose and by_component.get("unclassified"):
            out.write("## Unclassified keys (need new patterns)\n\n")
            for k in by_component["unclassified"]:
                out.write(f"- `{k}` {shapes[k]}\n")

    # Architectural answers
    with arch_md.open("w") as out:
        out.write("# Lance architecture — resolved from key inspection\n\n")
        for label, q in [
            ("Q1: MaPE Δ_m learned or hard-coded?", q1),
            ("Q2: LM head untied?", q2),
            ("Q3: Flow head structure?", q3),
            ("Q4: Attention QKV shared or split?", q4),
            ("Q5: Number of MaPE position groups?", q5),
        ]:
            out.write(f"## {label}\n\n")
            for k, v in q.items():
                out.write(f"- **{k}**: {v}\n")
            out.write("\n")

        out.write("\n## Next steps\n\n")
        out.write("1. Update `lance_mlx/model/lance_llm.py` LanceMoTLayer with the resolved attention topology (Q4)\n")
        out.write("2. Update `lance_mlx/model/flow_head.py` FlowHead with the resolved structure (Q3)\n")
        out.write("3. Update `lance_mlx/model/mape.py` MaPEOffsets `learned` flag based on Q1\n")
        out.write("4. Adjust `lance_mlx.model.routing.PositionGroup` enum if Q5 reveals a different group count\n")
        out.write("5. Read `llm_config.json` to confirm hyperparameters\n")

    print(f"Wrote: {full_txt}")
    print(f"Wrote: {summary_md}")
    print(f"Wrote: {arch_md}")
    print(f"\nTotal: {total_params / 1e9:.2f} B params, {len(shapes)} tensors")
    print(f"Components: {dict((c, len(by_component[c])) for c in sorted(by_component))}")
    print(f"\nVerdicts:")
    print(f"  Q1 MaPE Δ_m:       {q1['q1_verdict']}")
    print(f"  Q2 LM head:        {q2['q2_verdict']}")
    print(f"  Q3 Flow head:      {q3['q3_verdict']}")
    print(f"  Q4 Attn topology:  {q4['q4_verdict']}")
    print(f"  Q5 Position grps:  {q5['q5_verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
