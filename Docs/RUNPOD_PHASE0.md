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
| Instance | **A100 80GB PCIe, RunPod Community Cloud** |
| Template | PyTorch 2.4 (CUDA 12.4+) |
| Disk | ≥100 GB persistent volume |
| Wall-clock | ~2–3 hours (mostly the 60 GB HF download) |
| **Cost (realistic)** | **~$4–6 actual at ~$2/hr A100 80GB Community** |
| **Cost (budget w/ buffer)** | **~$15–20** (covers a re-run + accidental over-runs) |

## Cost breakdown (rounded, 2026 prices)

Pricing fluctuates; check runpod.io at booking time. Current ballpark for
A100 80GB:

- **RunPod Community Cloud:** $1.50–2.00 / hr (cheapest credible option for one-shot work). Spot-like SLA — fine for a single 2–3 hour session.
- **RunPod Secure Cloud:** $2.50–3.00 / hr. Higher SLA. Not needed here.
- **H100 80GB Community:** ~$3.00–4.00 / hr. ~2× faster than A100 for inference, but the speedup doesn't justify the cost for a one-shot oracle capture. Stick with A100.

Hourly × wall-clock estimate:

| Phase | Hours | Subtotal |
|---|---:|---:|
| Pod launch + env setup | 0.25 | $0.50 |
| HF download (60 GB unauthenticated) | 0.5–1.0 | $1.00–2.00 |
| 8–10 reference captures (T2I + T2V + edits + VQA) | 1.0–1.5 | $2.00–3.00 |
| scp out + terminate | 0.1 | $0.20 |
| **Total realistic** | **~2.5 hours** | **$4.50** |

**Add $10–15 of safety margin** for the inevitable "oh I forgot to capture X, let me re-launch" or "my prompt was wrong, redo case 3." A $20 RunPod credit covers everything comfortably with leftover.

> **Set an HF_TOKEN** before the download step — unauthenticated HF downloads are heavily rate-limited (often cuts a 60 GB pull from ~20 min to ~2 hours). Free token from https://huggingface.co/settings/tokens. This alone can save you $2–4 in pod time.

---

## Step-by-step

### 0. Before opening RunPod

- Have an HF token ready (free; https://huggingface.co/settings/tokens — read scope is fine).
- Have the Lance reference prompt set you want to capture (see `prompts/t2i_eval.json` etc. in this repo, or use the official examples in `bytedance/Lance/config/examples/`).
- Have an ssh public key handy if you prefer ssh over the web terminal (`cat ~/.ssh/id_ed25519.pub`).

### 1. Create the pod (~5 minutes)

1. Sign up / log in at https://runpod.io. Add ~$20 credit.
2. **Deploy → GPU Cloud → Community Cloud**.
3. Filter: **A100 80GB PCIe** (or SXM if PCIe out of stock — same price tier).
4. Template: **RunPod PyTorch 2.4** (or any PyTorch 2.x with CUDA ≥12.4).
5. Volume Disk: **100 GB** (60 GB weights + room for outputs + tmp).
6. Container Disk: 20 GB (default fine).
7. Optional: add your ssh pubkey under "Public Key".
8. **Deploy**. The pod takes 30–90 s to start.

### 2. Connect (web terminal or ssh)

In the pod page click **Connect**. Either:
- **Web Terminal** — easiest, no setup.
- **SSH** — copy the command shown; uses your registered pubkey.

### 3. Environment setup (~2 minutes)

```bash
cd /workspace
# HF token for fast, rate-limit-immune downloads
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Or: huggingface-cli login (interactive)

# Lance reference code
git clone https://github.com/bytedance/Lance.git
cd Lance
pip install -U pip
pip install -r requirements.txt   # CUDA 12.4+, flash-attn 2.6.3, triton 3.1.0
pip install -U "huggingface_hub[hf_transfer]"   # ~5× faster downloads
export HF_HUB_ENABLE_HF_TRANSFER=1
```

### 4. Download weights (~30–60 min depending on HF queue)

```bash
mkdir -p checkpoints
hf download bytedance-research/Lance --local-dir checkpoints/Lance
ls -la checkpoints/Lance/   # sanity: Lance_3B/, Lance_3B_Video/, Wan2.2_VAE.pth, Qwen2.5-VL-ViT/
du -sh checkpoints/Lance/   # ≈ 57 GB
```

### 5. Run the reference captures (~60–90 min total for 8–10 cases)

The shipped `inference_lance.sh` is the canonical entry point. Customize for
each test case (varying `--task`, `--prompt`, `--seed`, etc.). At minimum,
capture:

| # | task | resolution | frames | seed | steps | CFG | what |
|---|---|---|---|---|---|---|---|
| 1 | t2i | 768×768 | — | 42 | 30 | 4.0 | photoreal scene |
| 2 | t2i | 768×768 | — | 42 | 30 | 4.0 | stylized scene |
| 3 | t2i | 768×768 | — | 42 | 30 | 4.0 | edge case (text in image) |
| 4 | t2v | 480p | 50 | 42 | 30 | 4.0 | simple motion |
| 5 | t2v | 480p | 50 | 42 | 30 | 4.0 | complex motion |
| 6 | image_edit | 768×768 | — | 42 | 30 | 4.0 | "remove the X" |
| 7 | x2t_image | — | — | — | — | — | VQA on a held-out image |
| 8 | x2t_video | — | — | — | — | — | caption a held-out clip |

> **Use the same seeds in your MLX port.** This is the entire point of the
> oracle: byte-for-byte identical inputs produce comparable outputs.

For each case, save into `outputs/case_N/`:
- `prompt.json` — exact prompt + seed + CFG + step count
- `output.{png,mp4,txt}` — the generation
- `meta.txt` — wall-clock, peak VRAM, anything else from the script's stdout

Don't bother optimizing for speed here — you're paying for correctness, not
throughput.

### 6. scp back to your Mac (~5 min)

From your Mac (NOT inside the pod):

```bash
mkdir -p /Volumes/DEV_VOL1/VideoResearch/lance-mlx/tests/fixtures
# RunPod gives you an ssh string in the Connect tab — adapt:
scp -P <port> root@<pod-host>:/workspace/Lance/outputs/* \
  /Volumes/DEV_VOL1/VideoResearch/lance-mlx/tests/fixtures/
```

Or `tar czf oracle.tar.gz outputs/` on the pod first, scp that, untar locally.

### 7. **Terminate the pod**

This is the step people forget. The meter runs until you do.

In the RunPod console: **My Pods → ⋯ → Stop** then **Terminate**. (Stop alone
keeps the volume attached and continues billing the storage; Terminate fully
releases it.)

Verify your balance is decreasing as expected — should be ~$5 spent, not $50.

---

## Troubleshooting

- **`flash-attn` install fails.** It compiles against your CUDA version; the RunPod PyTorch 2.4 template ships matching CUDA so this usually works. If it doesn't, try `pip install flash-attn==2.6.3 --no-build-isolation`.
- **HF download stalls.** Confirm `HF_TOKEN` is exported and `HF_HUB_ENABLE_HF_TRANSFER=1`. Without these you're rate-limited and looking at 1–2 hours.
- **OOM during `t2v`.** A100 80GB should be enough at 50 frames / 480p; if you see OOM, drop `--num_frames` to 30 first to validate the pipeline, then increase.
- **Pod won't start.** Community Cloud A100s do go in-and-out of stock — try a different region (US-OR vs EU-CZ vs IN-NCR) or wait 10 minutes.

## After the capture

These fixtures become the parity oracle for every later phase of the MLX
port: each MLX-generated output is diffed against the matching PyTorch
fixture at the same seed/prompt/CFG. The handoff doc's Phase 2/3/4
validation gates all reference these.

Treat the captured fixtures as immutable — once they're in
`tests/fixtures/`, never regenerate them on a different seed or different
CFG, or every prior parity claim becomes incomparable.
