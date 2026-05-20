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
# Patterns reflect the 2026-05-19 verified findings: Lance's GEN expert uses a
# `_moe_gen` suffix on every attention/MLP/layernorm piece; UND is the bare base
# Qwen2.5-VL naming. Flow head is one Linear named `llm2vae`. Per-layer QK-norms
# live under self_attn.{q,k}_norm[/_moe_gen]. MaPE is hardcoded — should not
# appear in safetensors at all.
COMPONENT_PATTERNS = [
    # Embeddings + LM head — Lance wraps the LLM in `language_model.`
    (re.compile(r"(^|.*\.)embed_tokens(?:\.|$)"), "embeddings"),
    (re.compile(r"(^|.*\.)lm_head(?:\.|$)"), "lm_head"),

    # Bundled ViT — Lance_3B_Video bundles the Qwen2.5-VL ViT inside this
    # safetensors (Lance_3B does NOT). Claim ALL vit_model.* keys here as
    # one component, otherwise the ViT's mlp/norms leak into mlp_und/norm.
    (re.compile(r"^vit_model\.|^visual\.|^vit\.|^model\.visual\."), "vit"),

    # Lance GEN-expert keys — `_moe_gen` suffix. MUST match before base patterns.
    (re.compile(r".*\.q_proj_moe_gen\.|.*\.k_proj_moe_gen\.|.*\.v_proj_moe_gen\.|.*\.o_proj_moe_gen\."), "attn_moe_gen"),
    (re.compile(r".*\.q_norm_moe_gen(?:\.|$)|.*\.k_norm_moe_gen(?:\.|$)"), "qk_norm_moe_gen"),
    (re.compile(r".*\.mlp_moe_gen\."), "mlp_moe_gen"),
    (re.compile(r".*\.input_layernorm_moe_gen(?:\.|$)|.*\.post_attention_layernorm_moe_gen(?:\.|$)"), "layernorm_moe_gen"),
    # Final RMSNorm has a GEN sibling too (`norm_moe_gen`) — verified 2026-05-20.
    (re.compile(r"(^|.*\.)norm_moe_gen(?:\.|$)"), "final_norm_moe_gen"),

    # Lance flow head — LLM hidden → VAE velocity, named `llm2vae`.
    # NOTE: actual safetensors has both weight and bias (HANDOFF said bias=False; that was wrong).
    (re.compile(r"^llm2vae(?:\.|$)|.*\.llm2vae(?:\.|$)"), "flow_head"),
    # Symmetric input projection — VAE latent → LLM hidden, named `vae2llm`.
    # Verified 2026-05-20: NOT in original scaffolds.
    (re.compile(r"^vae2llm(?:\.|$)|.*\.vae2llm(?:\.|$)"), "vae_in_proj"),
    # Learned positional embedding over VAE latents (max_latent_size × max_latent_size grid).
    # Verified 2026-05-20: NOT in original scaffolds.
    (re.compile(r"^latent_pos_embed(?:\.|$)|.*\.latent_pos_embed(?:\.|$)"), "latent_pos_embed"),
    # Lance timestep embedder. Verified prefix: `time_embedder.mlp.{0,2}` (Sequential).
    (re.compile(r"(^|.*\.)time_embedder(?:\.|$)|(^|.*\.)t_embedder(?:\.|$)|(^|.*\.)time_embed(?:\.|$)"), "timestep_embedder"),

    # Base Qwen2.5-VL keys = Lance UND expert (no suffix at all).
    (re.compile(r".*\.self_attn\.q_proj\.|.*\.self_attn\.k_proj\.|.*\.self_attn\.v_proj\.|.*\.self_attn\.o_proj\."), "attn_und"),
    (re.compile(r".*\.self_attn\.q_norm(?:\.|$)|.*\.self_attn\.k_norm(?:\.|$)"), "qk_norm_und"),
    (re.compile(r".*\.mlp\.gate_proj\.|.*\.mlp\.up_proj\.|.*\.mlp\.down_proj\."), "mlp_und"),
    (re.compile(r".*\.input_layernorm(?:\.|$)|.*\.post_attention_layernorm(?:\.|$)"), "layernorm_und"),
    (re.compile(r".*\.self_attn\."), "attn_other"),

    # VAE — bundled separately as Wan2.2_VAE.pth; usually absent from LLM safetensors.
    (re.compile(r"^vae\.|^model\.vae\."), "vae"),
    # MaPE — verified hardcoded; should be ABSENT. Pattern here only catches surprises.
    (re.compile(r"^mape\.|^model\.mape_offsets|.*delta_m"), "mape"),
    # Connectors / multimodal projectors
    (re.compile(r"^connector\.|.*\.connector\.|^mm_projector\."), "connector"),

    # Final norm (UND side; GEN side handled above) + catch-all per-layer norms
    (re.compile(r"(^|.*\.)model\.norm(?:\.|$)|^final_norm(?:\.|$)"), "final_norm"),
    (re.compile(r".*\.norm(?:\.|$)"), "norm"),
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
    """Q3: What's the flow head structure?

    Verified 2026-05-19 (HANDOFF) said: single `nn.Linear(hidden_size, 48)` named
    `llm2vae`, `bias=False`. Empirical (2026-05-20) finding: weight AND bias are
    both present — the verified spec's `bias=False` claim was wrong (or the
    upstream code has `bias=True` overriding the docstring read). Scaffold must
    set `bias=True`.
    """
    fh_keys = by_component.get("flow_head", [])
    if not fh_keys:
        return {"q3_verdict": "UNKNOWN",
                "next_step": "Grep full dump for 'velocity', 'flow', 'gen_head', 'llm2vae'"}
    fh_shapes = {k: shapes[k] for k in fh_keys}
    keys_set = {k.split(".")[-1] for k in fh_keys}
    has_weight = any("llm2vae" in k and shapes[k][0] == 48 and len(shapes[k]) == 2 for k in fh_keys)
    if len(fh_keys) == 2 and keys_set == {"weight", "bias"} and has_weight:
        return {"q3_verdict": "SINGLE_LINEAR_48_WITH_BIAS",
                "shapes": fh_shapes,
                "evidence": "llm2vae has both weight and bias — scaffold should use bias=True",
                "next_step": "Update flow_head.py: nn.Linear(hidden, 48, bias=True)"}
    if len(fh_keys) == 1:
        k = fh_keys[0]
        shape = shapes[k]
        if "llm2vae" in k and len(shape) == 2 and shape[0] == 48:
            return {"q3_verdict": "VERIFIED_SINGLE_LINEAR_48_BIAS_FALSE",
                    "shape": shape,
                    "evidence": f"{k} shape={shape}, no bias"}
        return {"q3_verdict": "SINGLE_TENSOR_UNEXPECTED",
                "shape": shape,
                "next_step": "Verified flow head should be llm2vae shape=[48, hidden]. Investigate."}
    return {"q3_verdict": "MULTI_TENSOR_UNEXPECTED",
            "n_keys": len(fh_keys),
            "shapes": fh_shapes,
            "next_step": "Flow head was verified to be ONE Linear; >2-tensor here means the scaffold is wrong"}


def analyze_q4_attn(by_component: dict[str, list[str]]) -> dict:
    """Q4: Are attention QKV projections shared or duplicated?

    Verified 2026-05-19 + 2026-05-20: DUPLICATED. UND keeps the bare
    Qwen2.5-VL naming (`{q,k,v,o}_proj`); GEN adds `_moe_gen` siblings.

    Per-layer count: 7 attention tensors per expert per layer — 4 weights
    (q/k/v/o) + 3 biases (q/k/v; o_proj has no bias in Qwen2.5-VL). Times
    36 layers = 252 per side.
    """
    und = by_component.get("attn_und", [])
    gen = by_component.get("attn_moe_gen", [])
    n_layers_guess = 36
    expected_per_layer = 7   # 4 weights (q,k,v,o) + 3 biases (q,k,v); o_proj has no bias
    expected_total = n_layers_guess * expected_per_layer

    if und and gen:
        und_ok = (len(und) == expected_total)
        gen_ok = (len(gen) == expected_total)
        if und_ok and gen_ok:
            return {"q4_verdict": "DUPLICATED_MOE_GEN_VERIFIED",
                    "und_count": len(und),
                    "gen_count": len(gen),
                    "evidence": f"{len(und)} UND + {len(gen)} GEN attn keys = 36 layers × 7 tensors (4 weights + 3 biases) per side",
                    "next_step": "Confirm shape symmetry between matching (und, _moe_gen) pairs in convert.py"}
        return {"q4_verdict": "DUPLICATED_MOE_GEN_COUNT_DRIFT",
                "und_count": len(und),
                "gen_count": len(gen),
                "expected_per_side": expected_total,
                "next_step": f"Expected {expected_total} per side; investigate the delta"}
    if und and not gen:
        return {"q4_verdict": "UND_ONLY_NO_GEN_FOUND",
                "und_count": len(und),
                "evidence": "0 _moe_gen attention keys — either wrong checkpoint or different naming",
                "next_step": "Grep full dump for 'moe', 'gen', '_expert', or check Lance_3B vs Lance_3B_Video file"}
    return {"q4_verdict": "AMBIGUOUS",
            "und_count": len(und), "gen_count": len(gen),
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
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to a LOCAL model.safetensors")
    parser.add_argument("--remote-repo", default=None,
                        help="HuggingFace repo_id (e.g. bytedance-research/Lance) for REMOTE header-only mode")
    parser.add_argument("--remote-file", default=None,
                        help="Filename within --remote-repo (e.g. Lance_3B/model.safetensors)")
    parser.add_argument("--notes-dir", type=Path, default=Path("notes"))
    parser.add_argument("--name", default="lance")
    parser.add_argument("--verbose", action="store_true", help="Dump unclassified keys")
    args = parser.parse_args()

    if args.checkpoint is None and not (args.remote_repo and args.remote_file):
        print("ERROR: provide --checkpoint OR (--remote-repo + --remote-file)", file=sys.stderr)
        return 1
    if args.checkpoint and (args.remote_repo or args.remote_file):
        print("ERROR: --checkpoint is mutually exclusive with --remote-*", file=sys.stderr)
        return 1
    if args.checkpoint and not args.checkpoint.exists():
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

    if args.checkpoint:
        source_label = str(args.checkpoint)
        with safe_open(str(args.checkpoint), framework="numpy") as f, full_txt.open("w") as out:
            keys = list(f.keys())
            out.write(f"# {source_label}\n# {len(keys)} tensors\n\n")
            for key in keys:
                slice_ = f.get_slice(key)
                shape = list(slice_.get_shape())
                dtype = str(slice_.get_dtype())
                numel = 1
                for d in shape:
                    numel *= d
                total_params += numel
                dtypes[dtype] += 1
                shapes[key] = shape
                by_component[classify(key)].append(key)
                out.write(f"{key}\t{shape}\t{dtype}\t{numel:,}\n")
    else:
        from huggingface_hub import parse_safetensors_file_metadata
        source_label = f"hf://{args.remote_repo}/{args.remote_file}"
        print(f"Fetching safetensors header (header-only, ~KB) from {source_label} ...", file=sys.stderr)
        md = parse_safetensors_file_metadata(repo_id=args.remote_repo, filename=args.remote_file)
        with full_txt.open("w") as out:
            keys = list(md.tensors.keys())
            out.write(f"# {source_label}\n# {len(keys)} tensors (header-only)\n\n")
            for key in keys:
                ti = md.tensors[key]
                shape = list(ti.shape)
                dtype = str(ti.dtype)
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
        out.write(f"- Source: `{source_label}`\n")
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
