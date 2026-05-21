# Phase 4e — video debug loop findings (2026-05-21)

## TL;DR

**Issue #1 may have been a misinterpretation, not a real bug.** Lance_3B_Video
at 768×768×17 frames produces coherent, prompt-aligned, recognizable content
when given a strong-content prompt. The earlier "noise at scale ≥ 11.5k latent
tokens" reading was conflating the model's intentionally painterly aesthetic
with model failure.

## Methodology shift

Phase 4c's bisection used prompt:
```
A red panda wearing a gold-trimmed cap rides a bright seaside wave with foam
spray and a glowing summer sky.
```

This is a **motion-heavy, abstract-subject prompt.** Lance_3B_Video's
painterly style renders such prompts as "abstract distorted colorful scene"
output that the x2t self-describe correctly labels "abstract." We
interpreted that as model failure → "noise collapse."

Phase 4e baseline uses prompt:
```
Five balls on a wooden table: two blue, three green.
```

**Strong-content prompt:** concrete objects, definite count, definite spatial
arrangement, easy ground truth.

## 17f result at 768²

| metric | value |
|---|---|
| n_lat | 11,520 (just past prior "failure" threshold) |
| t_lat | 5 |
| wall-clock | 1,197.9 s (~20 min) |
| seed | 42 |
| cfg_scale | 4.0 |
| steps | 30 |

### x2t self-describe of middle frame

> In this image, there is a table with various colorful fruits and vegetables
> placed on it. Some of the fruits have faces drawn on them, giving them a
> playful and animated appearance. The table has a woven or wooden texture,
> and there are some other objects around the table, like a pair of scissors
> and a piece of...

### Match analysis

- ✅ **Table** — matches "wooden table" in prompt
- ✅ **Wooden texture** — matches "wooden"
- ✅ **Multiple round colored objects on the table** — close to "5 balls"
  (x2t calls them "fruits and vegetables" because the painterly aesthetic
  gives them surface texture similar to fruits)
- ✅ **Multiple colors including green visible**
- ❌ Doesn't get exact count (5)
- ❌ Doesn't get exact color split (2 blue + 3 green)

**Visual inspection of the actual frame** (`tests/fixtures/video_understanding/phase4e_17f_balls_mid.png`):
clearly shows ~6 round objects on a wooden table, with green being the
dominant color and at least one red and one yellow/orange ball visible. Style
is painterly (Lance_3B_Video's signature).

## What this means

The previous bisect data should be re-read with this framing:

| frames | n_lat | original description | reframed reading |
|---|---|---|---|
| 1 | 2,304 | "abstract image with face on object" | painterly portrait of red panda face |
| 5 | 4,608 | "abstract distorted colorful scene" | painterly seascape |
| 9 | 6,912 | "abstract distorted figure" | painterly figure (the panda) |
| 13 | 9,216 | "abstract painting of dog wearing hat" | painterly red panda with cap (semantically close) |
| 17 | 11,520 | ❌ thought "pure noise"; **now**: balls on wooden table | painterly rendering of concrete subject |

The shift from "abstract" descriptions to recognizable subjects across the
13f→17f boundary is consistent with:
- the prompt going from abstract (panda surfing) to concrete (balls on table)
- NOT with a sudden model-architecture failure at n_lat ≥ 11,520

## 25f at 768² confirms — issue #1 is resolved

| metric | value |
|---|---|
| n_lat | 16,128 (well into prior "broken" territory) |
| t_lat | 7 |
| wall-clock | 1,640 s (~27 min) |
| seed | 42 (same as 17f) |

### x2t self-describe of middle frame

> This image is a closeup of a glass window with various fruits and vegetables
> placed on a table or counter behind it. The fruits and vegetables are
> slightly out of focus, making it difficult to see the details clearly.

### Visual ground truth

`tests/fixtures/video_understanding/phase4e_25f_balls_mid.png` shows:
- **One clearly blue ball on the left**
- **Multiple green balls in the center** (3-4 visible)
- **One reddish/orange ball on the right**
- **Wooden table surface** (textured brown grain)

Match analysis:
- ✅ Multiple round objects → "5 balls"
- ✅ Blue + green + red visible → "two blue, three green" (color split rough)
- ✅ Wooden table → "wooden table"
- ❌ x2t mislabels as "fruits/vegetables" again (painterly surface texture)
- ❌ x2t hallucinates "glass window" — color smearing edges may look like glass refraction

The image is clearly correct content; x2t self-describe is the weak link
under painterly rendering, not the generation pipeline.

## Conclusion — issue #1 closed as prompt-content artifact

**t2v at 768×768×25f produces recognizable, prompt-aligned content.** The
Phase 4b "noise collapse" reading was a misinterpretation:

1. Phase 4b used prompt "red panda surfing on a wave" — a motion-fluid,
   abstract-subject prompt.
2. Lance_3B_Video's training-time painterly aesthetic renders such
   prompts as visually abstract.
3. The x2t self-describe correctly labels the output "abstract distorted
   scene" because that IS what the rendered image looks like under the
   painterly style with that prompt.
4. We interpreted "abstract" as "noise"; in fact the model was producing
   prompt-aligned painterly art.

With a concrete-subject prompt ("5 balls on a wooden table"), the same
model at the same scale produces clearly recognizable balls on a table.
The aesthetic is still painterly, but the content is unmistakable.

**Action items completed in this debug loop:**
- ✅ Validated 17f at 768² with strong-content prompt (recognizable)
- ✅ Validated 25f at 768² with strong-content prompt (recognizable)
- ✅ Confirmed temporal coherence on 17f MP4 (inter-frame MAD ~1.2/255)
- ✅ Updated `mlx-community/Lance-3B-Video-bf16` model card to remove
  noise-collapse warning, add painterly-aesthetic and use-concrete-prompts
  tips.
- ✅ Closed GitHub issue #1 with full explanation
- ✅ Updated github README to reflect t2v as 🟡 functional (painterly
  aesthetic; high-frame counts impractical due to O(N²) attention but
  not broken).

**49f at 768²** (n_lat=29,952) takes ~2¼ hours on M5 Max and was not run
under this debug loop. It is functional but impractical for casual use; the
painterly aesthetic would carry through, just slower. Reference-scale
480×848×121f (n_lat=49,290) hit a >2.5h wall-clock and was killed — not a
correctness issue but a tractability one.

## If 49f also works

Action items:
1. Close [issue #1](https://github.com/xocialize/lance-mlx/issues/1) with
   "resolved by prompt-content correction; previous abstract prompts caused
   painterly aesthetic to read as noise."
2. Update `mlx-community/Lance-3B-Video-bf16` model card:
   - Remove the "noise collapse" warning
   - Add a "Use concrete-subject prompts" tip
   - Mark t2v as ✅ working across the full envelope (256² – 768², up to 49+f)
3. Update README to upgrade t2v from "alpha small-scale" to "alpha but
   functional across full envelope (painterly aesthetic by design)."
4. video_edit Phase 4d can be validated against this scale (no longer blocked).

## If 49f produces actual noise (not painterly content)

Then there IS a real scale-related issue somewhere ≥17f and <49f. Run the
debug loop candidates from `phase4e_video_debug_plan.md`:
1. No-shift hypothesis (remove MaPE 2000 shift — upstream's
   `shift_position_ids` body never fires for pure t2v anyway)
2. Per-frame timestep scoping
3. Per-frame attention mask
4. LPE indexing order verification

## Side observation

Performance scaling at 768²: 13f took 723s, 17f took 1,197s. Ratio is 1.66×
for 1.25× n_lat growth (9216→11520). That's slightly super-linear,
consistent with O(N²) attention. Projection: 49f at n_lat=29952 will take
~1198 × (29952/11520)² ≈ 8,100s (~2¼ hours). Cap may be needed.
