# Phase 0 — Parity-Oracle Capture on RunPod (runbook)

Goal: produce reference PyTorch outputs from ByteDance's official Lance
inference at fixed seeds, so the MLX port has something objective to diff
against in every later phase. Capture, scp back, terminate the box, port
locally.

The Lance reference code is CUDA-only (`assert torch.cuda.is_available()` is
the first line of `main()`), so this is the one phase that *must* run on a
rented cloud GPU. After this capture, nothing else in the project leaves
your Mac.

---

## TL;DR

| Item | Detail |
|---|---|
| Instance | **A100 80GB PCIe** ($1.39/hr) or **A100 80GB SXM** ($1.49/hr) — Community Cloud |
| Template | RunPod PyTorch 2.8 (CUDA 12.8) — anything ≥ PyTorch 2.4 / CUDA 12.4 works |
| Storage | **120 GB container disk + 10 GB network volume** (ephemeral; nothing needs to persist past termination) |
| Auth | SSH public key pasted into the deploy form (uncheck Jupyter — not needed) |
| Wall-clock | **~1.5–2 hours** with the narrowed download below (HF_TOKEN + hf_transfer essential) |
| **Cost (realistic)** | **~$3–5 actual** at $1.40–1.50/hr |
| **Cost (budget w/ buffer)** | **~$10–15** (covers a re-run + accidental over-runs) |

## Cost breakdown (rounded, 2026 prices)

Pricing fluctuates; check runpod.io at booking time. Current ballpark for
A100 80GB:

- **RunPod Community Cloud:** $1.50–2.00 / hr (cheapest credible option for one-shot work). Spot-like SLA — fine for a single 2–3 hour session.
- **RunPod Secure Cloud:** $2.50–3.00 / hr. Higher SLA. Not needed here.
- **H100 80GB Community:** ~$3.00–4.00 / hr. ~2× faster than A100 for inference, but the speedup doesn't justify the cost for a one-shot oracle capture. Stick with A100.

Hourly × wall-clock estimate (with the **narrowed download** below):

| Phase | Hours | Subtotal |
|---|---:|---:|
| Pod launch + env setup | 0.25 | $0.40 |
| HF download (**~33 GB**, authenticated + hf_transfer) | 0.15–0.30 | $0.20–0.45 |
| 8–10 reference captures (T2I + T2V + edits + VQA) | 1.0–1.5 | $1.50–2.25 |
| scp out + terminate | 0.1 | $0.15 |
| **Total realistic** | **~1.5–2 hours** | **$2.25–3.25** |

**Add $7–10 of safety margin** for the inevitable "oh I forgot to capture X, let me re-launch" or "my prompt was wrong, redo case 3." Your existing $25 credit covers everything comfortably with plenty of leftover.

> **Set an HF_TOKEN** before the download step — unauthenticated HF downloads are heavily rate-limited (turns the narrowed ~33 GB pull from ~5–10 min into ~30–60 min). Free token from https://huggingface.co/settings/tokens. This single env var is the biggest cost/time lever in the whole run.

---

## Step-by-step

### 0. Before opening RunPod

- **HF token ready** (free; https://huggingface.co/settings/tokens — read scope is fine). Skipping this turns a 5-min download into a 90-min one.
- **SSH public key** ready to paste — `cat ~/.ssh/id_ed25519.pub` on your Mac. This enables both SSH-in (better than web terminal — tmux works, multiple sessions) and the scp-back step at the end with no password prompts.
- (Optional) The Lance reference prompt set you want to capture — the official examples in `bytedance/Lance/config/examples/` are fine.

### 1. Create the pod (~5 minutes)

1. Sign up / log in at https://runpod.io. Add $20+ credit.
2. **Deploy → GPU Cloud → Community Cloud**.
3. Filter: **A100 80GB PCIe** ($1.39/hr) or **A100 80GB SXM** ($1.49/hr) — SXM has been more reliably in stock recently.
4. Template: **RunPod PyTorch 2.8** (CUDA 12.8) — newer than the minimum, fully fine.
5. Storage → **Container disk: 120 GB**, **Network volume: 10 GB** (the minimum; we don't need persistence — terminate at the end).
6. **Paste your SSH public key** in the SSH key field. **Uncheck "Start Jupyter notebook"** (not needed; saves boot time + RAM).
7. **Deploy**. The pod takes 30–90 s to start. If "Out of capacity" — try a different region in the dropdown, or wait 10–30 min.

### 2. Connect via SSH (~30 s)

In the pod page click **Connect**. Use the **SSH** tab — copy the `ssh root@... -p NNNNN -i ~/.ssh/id_ed25519` command shown and paste it into a terminal on your Mac. First connection asks "Are you sure you want to continue connecting? [yes]" — type `yes`. You should see a `#` prompt.

(The Web Terminal also works but is fragile — disconnects on browser refresh and doesn't survive a closed laptop lid. SSH does.)

### 2a. Sanity-check (~10 s)

```bash
nvidia-smi             # should show an A100 80GB; if not, the pod was misprovisioned
df -h /                # should show ~120 GB available
df -h /workspace       # should show ~10 GB available (network volume)
```

### 3. Environment setup (~2 minutes)

**Start a tmux session first** so the SSH connection dropping (laptop sleeping, wifi blip, accidental window close) doesn't kill your download or capture run:

```bash
tmux new -s phase0           # creates a session named "phase0"
# If you ever get disconnected, reconnect via ssh, then: tmux attach -t phase0
```

Inside the tmux session:

```bash
cd /                                                         # NOT /workspace — that's the 10 GB volume
mkdir -p /lance && cd /lance                                 # work area on the 120 GB container disk

# HF token for fast, rate-limit-immune downloads
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx          # paste yours; READ scope is fine
export HF_HUB_ENABLE_HF_TRANSFER=1                           # ~5× faster downloads

# Lance reference code
git clone https://github.com/bytedance/Lance.git
cd Lance
pip install -U pip
pip install -r requirements.txt                              # flash-attn, triton, etc.
pip install -U "huggingface_hub[hf_transfer]"
```

> Why `/lance/Lance` instead of `/workspace/Lance`: `/workspace` is the 10 GB network volume (too small for 33 GB of weights). The 120 GB container disk is rooted at `/`. We bypass `/workspace` entirely for this one-shot run.

### 4. Download weights — narrowed pull (~5–15 min with hf_transfer)

`Lance_3B_Video` handles **all 6 tasks** per `inference_lance.sh` defaults
(both checkpoints share `llm_config.json` byte-for-byte, verified). Skip
`Lance_3B` (~24.7 GB image-only variant); we don't need it.

```bash
mkdir -p downloads

hf download bytedance-research/Lance \
  --include "Lance_3B_Video/*" \
  --include "Wan2.2_VAE.pth" \
  --include "Qwen2.5-VL-ViT/*" \
  --include "LICENSE" "README.md" \
  --local-dir downloads
```

Expected output:

```bash
ls -la downloads/                  # Lance_3B_Video/, Wan2.2_VAE.pth, Qwen2.5-VL-ViT/, LICENSE, README.md
du -sh downloads/                  # ~33 GB
du -sh downloads/Lance_3B_Video    # ~28.4 GB
du -sh downloads/Wan2.2_VAE.pth    # ~2.8 GB
du -sh downloads/Qwen2.5-VL-ViT    # ~1.3 GB
```

> If `hf` isn't found, the modern CLI is the `hf` command from the newer `huggingface_hub` package; an older `huggingface-cli` may be on PATH instead. Either of these works:
> ```bash
> huggingface-cli download bytedance-research/Lance --include "Lance_3B_Video/*" ...
> # or just python -c "from huggingface_hub import snapshot_download; snapshot_download('bytedance-research/Lance', allow_patterns=['Lance_3B_Video/*','Wan2.2_VAE.pth','Qwen2.5-VL-ViT/*'], local_dir='downloads')"
> ```

### 5. Run the reference captures (~60–90 min total for 8 cases)

> **Read `notes/phase0_config_factory.md` first** — the pre-flight dump of every
> dataclass default + shell override. It documents the canonical run command,
> the chat templates, and three load-bearing surprises (`vae_model_type` default
> `seedance` vs shell `wan`; `latent_patch_size` default `[1,2,2]` vs shell
> `[1,1,1]`; `vit_type` default `qwen2_5_vl` vs shell `qwen_2_5_vl_original`).
> All of those values are baked into `inference_lance.sh` already — you just need
> to vary `TASK_NAME`, `RESOLUTION`, and the input prompt per case.

The shipped `inference_lance.sh` is the canonical entry point but it expects
**in-file editing** of `TASK_NAME` / `RESOLUTION` / `NUM_FRAMES` etc. — there are
no CLI overrides for these. The clean way to do this without losing track of
which case ran with which flags is to **make per-case copies of the script**:

```bash
cd /lance/Lance
# Make 8 copies, one per case — edit each one's top-of-file params before running.
for n in 01_t2i_photoreal 02_t2i_stylized 03_t2i_text \
         04_t2v_simple 05_t2v_complex \
         06_image_edit 07_x2t_image 08_x2t_video; do
  cp inference_lance.sh phase0_${n}.sh
done
ls phase0_*.sh
```

Each per-case script edits the **first ~25 lines** of `inference_lance.sh` —
those are the user-facing knobs. Use a text editor (`nano`, `vim`, or
`code-server` if you installed it). Recommended capture set:

| # | Edit `TASK_NAME=` | Edit `RESOLUTION=` | `NUM_FRAMES=` | what to capture |
|---|---|---|---|---|
| 1 | `t2i` | `image_768res` | (unused) | photoreal scene (e.g. fox prompt) |
| 2 | `t2i` | `image_768res` | (unused) | stylized scene |
| 3 | `t2i` | `image_768res` | (unused) | edge case (text rendering in image) |
| 4 | `t2v` | `video_480p` | `50` | simple motion (single subject) |
| 5 | `t2v` | `video_480p` | `50` | complex motion (multiple objects) |
| 6 | `image_edit` | `image_768res` | (unused) | "remove the X" example from config/examples/ |
| 7 | `x2t_image` | `video_480p` | (unused) | VQA on a held-out image |
| 8 | `x2t_video` | `video_480p` | (unused) | caption a held-out clip |

Leave `VALIDATION_DATA_SEED=42`, `VALIDATION_TIMESTEP_SHIFT=3.5`,
`CFG_TEXT_SCALE=4.0`, `USE_KVCACHE=true` **unchanged** across all 8 — those are
the parity baseline. Also leave `MODEL_PATH="downloads/Lance_3B_Video"`.

Run each:

```bash
bash phase0_01_t2i_photoreal.sh 2>&1 | tee logs/01.log
# ... and so on
```

The script auto-creates `results/<task>_sample_ts30_tts3.5_seed42_cfg4.0_kvcache_<timestamp>/`
per run. After all 8 are done:

```bash
ls -la results/                            # 8 timestamped directories
du -sh results/                            # total fixture footprint, expect 1–3 GB
```

> **Use the same seeds in your MLX port later.** This is the entire point of the
> oracle: byte-for-byte identical inputs produce comparable outputs.

Don't optimize for speed — you're paying for correctness, not throughput.

### 6. scp back to your Mac (~5 min)

**Step 1, on the pod**: tar everything into one file for a single scp call:

```bash
cd /lance/Lance
tar czf /tmp/phase0_oracle.tar.gz results/ logs/
ls -lh /tmp/phase0_oracle.tar.gz                  # note the size — ~1–3 GB
```

**Step 2, on your Mac** (open a new terminal — *not* inside the SSH session): find your pod's SSH info. In the RunPod console → your pod's page → **Connect → SSH** tab, the command shown looks like `ssh root@123.45.67.89 -p 12345 -i ~/.ssh/id_ed25519`. You need the host (`123.45.67.89`) and port (`12345`) from that string.

```bash
mkdir -p /Volumes/DEV_VOL1/VideoResearch/lance-mlx/tests/fixtures
scp -P <port> -i ~/.ssh/id_ed25519 \
    root@<host>:/tmp/phase0_oracle.tar.gz \
    /Volumes/DEV_VOL1/VideoResearch/lance-mlx/tests/fixtures/
```

If you minted a dedicated `id_ed25519_runpod` key earlier, swap that in for `-i`.

**Step 3, on your Mac**: unpack and verify:

```bash
cd /Volumes/DEV_VOL1/VideoResearch/lance-mlx/tests/fixtures
tar xzf phase0_oracle.tar.gz
ls -la results/                                   # 8 timestamped dirs
```

Once you've confirmed the fixtures are present on the Mac and readable, you can move on to terminate the pod.

### 7. **Terminate the pod**

This is the step people forget. The meter runs until you do.

In the RunPod console: **My Pods → ⋯ → Stop** then **Terminate**. (Stop alone
keeps the volume attached and continues billing the storage; Terminate fully
releases it.)

Verify your balance is decreasing as expected — should be **~$3 spent, not $30**. If you see a much higher charge, double-check the pod is fully terminated and the network volume is gone too (network volumes can keep billing storage if left attached).

---

## Troubleshooting

- **`flash-attn` install fails.** Compiles against your CUDA version; the RunPod PyTorch 2.8 / CUDA 12.8 template normally just works. If it doesn't: `pip install flash-attn==2.6.3 --no-build-isolation`.
- **HF download stalls or is slow.** Confirm `echo $HF_TOKEN` and `echo $HF_HUB_ENABLE_HF_TRANSFER` both show values. Without auth + hf_transfer you're rate-limited and looking at hours instead of minutes.
- **OOM during `t2v`.** A100 80GB should be enough at 50 frames / 480p; if you see OOM, drop `NUM_FRAMES=` to 30 in the script first to validate the pipeline, then increase.
- **Pod won't start ("out of capacity").** Community Cloud A100s go in-and-out of stock — try **A100 SXM** if PCIe is out, or H100 PCIe (~$2.89/hr) as a faster substitute, or just wait 10–30 min.
- **SSH connection dropped mid-download.** This is exactly why we started inside `tmux new -s phase0`. Reconnect via SSH, then `tmux attach -t phase0` to resume the session right where you left off. If you forgot to use tmux, the download is dead and needs to restart.
- **`hf: command not found`** — try `huggingface-cli` (older CLI name). Both ship in the modern `huggingface_hub` package.

## After the capture

These fixtures become the parity oracle for every later phase of the MLX
port: each MLX-generated output is diffed against the matching PyTorch
fixture at the same seed/prompt/CFG. The handoff doc's Phase 2/3/4
validation gates all reference these.

Treat the captured fixtures as immutable — once they're in
`tests/fixtures/`, never regenerate them on a different seed or different
CFG, or every prior parity claim becomes incomparable.

**Next step:** Phase 1a — weight inspection. Run `scripts/01_inspect_keys.py`
against the downloaded `Lance_3B_Video/model.safetensors` (on your Mac now,
locally — no more cloud needed). This enumerates the actual tensor names so
the converter knows what to map. The handoff's "⚠ Verified findings" section
covers everything else still open.

## If you hit something not covered

Run on the pod (BEFORE terminating):

```bash
# print the env + tool versions for context
echo "torch: $(python -c 'import torch; print(torch.__version__)')"
echo "cuda:  $(nvcc --version | tail -1)"
echo "hf:    $(hf --version 2>&1 || huggingface-cli --version 2>&1)"
nvidia-smi -L

# capture the failing command's full output
bash phase0_<n>_<task>.sh 2>&1 | tee logs/<n>_repro.log
```

Then ping me with the log content. Don't terminate the pod while you're
asking — the meter cost of an idle A100 for 30 min is ~$0.75; the cost of
re-launching, re-downloading 33 GB, and re-running setup is closer to $1.50
plus a lot more of your time.
