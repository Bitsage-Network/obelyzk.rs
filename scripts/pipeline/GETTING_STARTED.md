# Getting Started with obelyzk.rs

Prove that an ML model ran correctly, and verify the proof on-chain. Works on any NVIDIA GPU.

---

## What You Need

- A machine with an NVIDIA GPU (RTX 4090, A100, H100, H200, B200, or B300)
- Ubuntu/Debian or RHEL/Rocky Linux
- Internet connection
- ~50GB free disk space (for model + build + proof)

**Optional:**
- A HuggingFace token (only for gated models like Llama or Gemma)
- A Starknet wallet private key (only if you want to pay your own gas — Sepolia uses free AVNU paymaster by default)

---

## The Fastest Way (One Command)

SSH into your GPU machine and run:

```bash
# Single-prompt inference + ZK proof + on-chain verification
curl -fsSL https://raw.githubusercontent.com/Bitsage-Network/obelyzk.rs/main/scripts/pipeline/bootstrap.sh | bash -s -- --preset qwen3-14b --gpu --submit
```

**Multi-turn conversation** — have the model discuss a topic across multiple turns, with each turn ZK-proved:

```bash
# 3-turn conversation about quantum computing, each turn proved + verified on-chain
curl -fsSL https://raw.githubusercontent.com/Bitsage-Network/obelyzk.rs/main/scripts/pipeline/bootstrap.sh | bash -s -- --preset qwen3-14b --gpu --submit --conversation-topic "quantum computing"

# 5-turn deep dive
curl -fsSL https://raw.githubusercontent.com/Bitsage-Network/obelyzk.rs/main/scripts/pipeline/bootstrap.sh | bash -s -- --preset qwen3-14b --gpu --submit --conversation-topic "cryptography and post-quantum security" --conversation-turns 5
```

Or clone and run manually:

```bash
git clone https://github.com/Bitsage-Network/obelyzk.rs.git
cd obelyzk.rs/scripts/pipeline

# Dry run (no on-chain submission) — good for first try
./run_e2e.sh --preset phi3-mini --gpu --dry-run

# Single prompt + on-chain
./run_e2e.sh --preset qwen3-14b --gpu --submit

# Multi-turn conversation + on-chain
./run_e2e.sh --preset qwen3-14b --gpu --submit --conversation-topic "neural networks"
```

This does everything: installs drivers, downloads the model, runs inference (single prompt or multi-turn conversation), captures real inference logs through the proved forward pass, generates a ZK proof for each entry, verifies on-chain, and runs an audit.

During setup, you'll be prompted for your email. This links your device to your [marketplace dashboard](https://marketplace.bitsage.network) where you can view all your proofs and audit reports. If you don't have an account yet, sign up at `marketplace.bitsage.network/signup` with the same email after the pipeline completes.

Gas is paid by Obelysk via AVNU paymaster. To use your own account instead:

```bash
STARKNET_PRIVATE_KEY=0x_your_key ./run_e2e.sh --preset qwen3-14b --gpu --submit --no-paymaster
```

That's it. The rest of this guide explains what each step does and how to run them individually.

---

## Step-by-Step Guide

### Step 0 — Set Up the GPU Machine

**What it does:** Installs system packages, NVIDIA drivers, CUDA, Rust, and builds the prover binary. Also builds llama.cpp for inference testing.

```bash
./00_setup_gpu.sh
```

When it finishes, your GPU is detected and ready. The script saves its config to `~/.obelysk/gpu_config.env`.
It also writes CUDA shell env to `~/.obelysk/cuda_env.sh`.

If setup stops with `Driver/library version mismatch`, that means the driver was updated and a reboot is required:

```bash
sudo reboot
# after reconnect
cd ~/obelyzk.rs/scripts/pipeline
./00_setup_gpu.sh --skip-deps
```

To use `nvcc` in your current shell after setup:

```bash
source ~/.obelysk/cuda_env.sh
nvcc --version
```

**Common flags:**
| Flag | What it does |
|------|--------------|
| `--install-drivers` | Force install NVIDIA drivers + CUDA |
| `--skip-drivers` | Skip driver install (already installed on cloud instances) |
| `--skip-build` | Skip building Rust binaries (already built) |
| `--skip-llama` | Skip building llama.cpp |

If you plan to skip inference testing (`run_e2e.sh --skip-inference`), llama.cpp build is now skipped automatically.

**Example — cloud instance with pre-installed drivers:**
```bash
./00_setup_gpu.sh --skip-drivers
```

---

### Step 1 — Download a Model

**What it does:** Downloads a model from HuggingFace and saves it to `~/.obelysk/models/`.

```bash
# Pick a preset
./01_setup_model.sh --preset phi3-mini        # 7GB, fastest for testing
./01_setup_model.sh --preset qwen3-14b        # 28GB, production default
./01_setup_model.sh --preset mistral-7b       # 15GB

# See all presets
./01_setup_model.sh --list
```

**Available presets:**

| Preset | Size | Needs HF Token? |
|--------|------|-----------------|
| `phi3-mini` | 7GB | No |
| `mistral-7b` | 15GB | No |
| `llama3-8b` | 16GB | Yes |
| `gemma2-9b` | 18GB | Yes |
| `qwen3-14b` | 28GB | No |
| `llama3-70b` | 140GB | Yes |

**For gated models (Llama, Gemma):**
1. Go to https://huggingface.co/settings/tokens and create a token
2. Accept the model's license on its HuggingFace page
3. Run:
```bash
HF_TOKEN=hf_your_token ./01_setup_model.sh --preset llama3-8b
```

**Custom model (not in presets):**
```bash
./01_setup_model.sh --hf-model Qwen/Qwen3-0.5B --layers 24
```

---

### Step 2 — Validate the Model

**What it does:** Checks that all model files downloaded correctly, dimensions match, and weights are valid.

```bash
./02_validate_model.sh
```

This runs automatically. No flags needed.

`run_e2e.sh` uses this real validation path by default and does not run a separate 1-layer debug proof unless `--validate-test-proof` is explicitly set.

---

### Step 2a — Test Inference (Optional)

**What it does:** Converts the model to GGUF format and runs it through llama.cpp to confirm it produces real output. This is optional but recommended to verify the model works before proving.
If `torch` is missing, the script auto-installs it for conversion and streams install/conversion output live.
Use `INFERENCE_TIMEOUT_SEC` (default `900`) to control prompt/benchmark timeout.
`Ctrl-C` now terminates inference runs cleanly (no stuck subprocess).

```bash
# Quick test — ask it a question
./02a_test_inference.sh --model-name phi3-mini

# Custom prompt
./02a_test_inference.sh --model-name phi3-mini --prompt "Explain gravity in one sentence"

# Interactive chat
./02a_test_inference.sh --model-name phi3-mini --chat

# Speed benchmark
./02a_test_inference.sh --model-name phi3-mini --benchmark
```

---

### Step 2b — Capture Inference Log (Required)

**What it does:** Runs the model through the prover's forward pass (`execute_forward_pass()`) and records each inference in a chain-linked log. This is the same code path the audit verifier checks, ensuring the log is genuine. The capture step is mandatory before proving/auditing.

```bash
# Default: 3 forward passes with diverse random input
./02b_capture_inference.sh

# Real text prompt — tokenizes, extracts embedding, proves real inference
./02b_capture_inference.sh --prompt "Derive the eigenvalues of the 3x3 rotation matrix"

# Fewer captures (faster, for testing)
./02b_capture_inference.sh --count 5

# Specify layers (for partial model capture)
./02b_capture_inference.sh --layers 1
```

**Using `--prompt`:** When you pass `--prompt "text"`, the binary:
1. Loads `tokenizer.json` from the model directory
2. Tokenizes the prompt and takes the last token ID
3. Extracts that token's embedding row directly from the SafeTensors weight file (~40 KB, not the full ~3 GB table)
4. Uses that real embedding as input for all forward passes

The proof then covers a real transformer computation on real data — not random numbers. The M31 output is modular arithmetic and cannot be decoded to text (text decoding is a future release).

When using `run_e2e.sh --submit`, a default mathematical prompt is set automatically. Use `--no-prompt` to disable this and use random input instead.

The log is saved to `~/.obelysk/logs/<model_name>/` and contains:
- `meta.json` — session metadata (model ID, weight commitment)
- `log.jsonl` — chain-linked inference entries (each entry hashes the previous)
- `matrices.bin` — M31 matrix data for input/output replay

`02b_capture_inference.sh` now streams capture output live with a 15s heartbeat.
Use `CAPTURE_TIMEOUT_SEC` (default `3600`) to bound capture runtime.

---

### Step 2c — Multi-Turn Conversation Capture (Optional)

**What it does:** Generates a multi-turn conversation about a topic using the real model (float16 via HuggingFace), then proves each turn through the M31 forward pass. This produces an inference log where each entry has real text responses — not just M31 output.

**Fastest way (bootstrap one-liner):**
```bash
curl -fsSL https://raw.githubusercontent.com/Bitsage-Network/obelyzk.rs/main/scripts/pipeline/bootstrap.sh | bash -s -- --preset qwen3-14b --gpu --submit --conversation-topic "quantum computing"
```

Two-phase approach:
1. **Float path** (Python): Loads the model in float16, generates real text responses autoregressively
2. **Proof path** (Rust): For each turn, extracts the last token's embedding and runs the M31 forward pass

**Standalone:**
```bash
# 3-turn conversation about quantum computing
./02c_conversation.sh --topic "quantum computing and its implications for cryptography" --turns 3

# Quick 2-turn test
./02c_conversation.sh --topic "prime numbers" --turns 2

# Use a pre-generated conversation.json (skip Python inference)
./02c_conversation.sh --conversation-file /tmp/my_conversation.json

# With specific layers
./02c_conversation.sh --topic "neural networks" --turns 2 --layers 1
```

**Full pipeline (E2E):**
```bash
# E2E with multi-turn conversation + on-chain proof
./run_e2e.sh --preset qwen3-14b --gpu --submit --conversation-topic "quantum computing"

# More turns
./run_e2e.sh --preset qwen3-14b --gpu --submit --conversation-topic "cryptography" --conversation-turns 5
```

**GPU memory note:** The float model (~28GB for Qwen3-14B) and M31 prove-model both need GPU memory. They run sequentially — Python generates all turns first, then Rust proves them all.

The conversation log entries are chain-linked just like regular captures. Each entry includes the real model response text in `output_preview` and the generated token IDs in `output_tokens`. The normal prove → verify → on-chain flow handles all entries.

**conversation.json format:** The intermediate file contains the topic, model info, and per-turn data (user query, full context tokens, last token ID, response text, response tokens, timing).

---

### Step 3 — Generate the Proof

**What it does:** Runs the model through the prover, which generates a cryptographic proof that the inference was computed correctly. Then verifies the proof locally before saving it.

```bash
# Default mode (GKR) — fastest
./03_prove.sh --model-name qwen3-14b --gpu

# Prove just 1 layer (faster, good for testing)
./03_prove.sh --model-name qwen3-14b --layers 1 --gpu

# Multi-GPU (if you have multiple GPUs)
./03_prove.sh --model-name qwen3-14b --gpu --multi-gpu

# Enforce GPU-only proving (fail fast on CPU fallback)
./03_prove.sh --model-name qwen3-14b --gpu --gpu-only
```

**Proof modes:**

| Mode | Speed | Security | Command |
|------|-------|----------|---------|
| `gkr` | Fastest | High | `--mode gkr` (default) |
| `direct` | Moderate | High | `--mode direct` |
| `recursive` | Slowest | Highest | `--mode recursive` |

The proof is saved to `~/.obelysk/proofs/`.

**Important runtime behavior (qwen3-14b):**
- First run computes and caches a model weight commitment (typically a few minutes).
- Later runs reuse the cache immediately if weights are unchanged.
- Phase 2 has two subphases:
  - GKR layer reductions (usually fast, tens of seconds)
  - Weight opening proofs (can dominate runtime)
- Opening prep now avoids an extra full padded-matrix copy, reducing CPU RAM pressure during large openings.
- GPU opening trees now use device-side QM31 packing, removing per-round CPU repack/upload overhead in large weight openings.
- Opening query extraction now replays folds on GPU and downloads only queried leaf pairs, avoiding full-layer per-round downloads.
- Unified STARK retries once on SIMD if GPU proving returns `ConstraintsNotSatisfied` (soundness-preserving fallback). In `--gpu-only` mode this fallback is disabled and the run fails closed.
- `03_prove.sh` now defaults to `STWO_PURE_GKR_SKIP_UNIFIED_STARK=1` for `ml_gkr` runs, skipping Phase 3 when GKR already covers non-matmul components.

The pipeline now prints dense progress + heartbeat messages during long openings so it does not appear frozen.

If logs show:
```text
[GKR] MLE Merkle backend: CPU fallback (...)
```
then opening commitments are no longer fully GPU-resident and runtime will increase sharply.
Use GPU-only fail-fast mode to catch this immediately:
```bash
./03_prove.sh --model-name qwen3-14b --gpu --gpu-only
```

**Default fast-safe tuning (no soundness downgrade):**
- `STWO_GPU_COMMIT_STRICT` and `STWO_GPU_COMMIT_HARDEN` stay off unless you explicitly set them.
- GPU fold is enabled by default for large MLE openings:
  - `STWO_GPU_MLE_FOLD=1`
  - `STWO_GPU_MLE_FOLD_MIN_POINTS=1048576`
- GPU opening-tree path is enabled by default to avoid bulk Merkle layer downloads:
  - `STWO_GPU_MLE_OPENING_TREE=1`
  - (with `--gpu-only`, pipeline also enforces `STWO_GPU_MLE_OPENING_TREE_REQUIRE=1`)
- Optional timing debug for opening internals:
  - `STWO_GPU_MLE_OPENING_TIMING=1`
- CPU fallback sections use all CPU threads by default:
  - `RAYON_NUM_THREADS=$(nproc)`
  - `OMP_NUM_THREADS=$(nproc)`

**Useful knobs:**
```bash
# Fail if critical paths fall back to CPU
./03_prove.sh --model-name qwen3-14b --gpu --gpu-only

# Keep strict/hardening checks explicitly enabled (slower)
export STWO_GPU_COMMIT_STRICT=1
export STWO_GPU_COMMIT_HARDEN=1
export STWO_GPU_POLY_STRICT=1
export STWO_GPU_POLY_HARDEN=1
# Force fail-closed instead of Unified STARK GPU->SIMD fallback
export STWO_UNIFIED_STARK_NO_FALLBACK=1
# Re-enable Phase 3 unified STARK in pure GKR mode (default is skip)
export STWO_PURE_GKR_SKIP_UNIFIED_STARK=0
./03_prove.sh --model-name qwen3-14b --gpu
```

**Weight-binding mode defaults (current):**
- `--submit` is the canonical path and defaults to **`verify_model_gkr_v4` mode 4**
  (aggregated oracle sumcheck, submit-ready, ~17K felts calldata).
- `--starknet-ready` is deprecated and now aliases to `--submit` unless you pass
  explicit legacy selectors.
- Legacy selectors (`--gkr-v2`, `--gkr-v3`, `--gkr-v4-mode3`, `--legacy-gkr-v1`)
  remain available for compatibility/testing only.

Recommended command:
```bash
./03_prove.sh --model-name qwen3-14b --gpu --submit
```

Legacy compatibility examples:
```bash
# Legacy v1 sequential-opening path
./03_prove.sh --model-name qwen3-14b --gpu --submit --legacy-gkr-v1

# Legacy v4 mode3 envelope (explicit legacy selector)
./03_prove.sh --model-name qwen3-14b --gpu --starknet-ready --gkr-v4-mode3
```

---

### Step 4 — Verify On-Chain

**What it does:** Submits the proof to Starknet. On Sepolia, gas is paid by
Obelysk via AVNU paymaster — you don't need STRK or a wallet.

```bash
# Dry run first — shows what will be submitted without sending anything
./04_verify_onchain.sh --dry-run

# Zero-config (Sepolia) — just works, no setup needed
./04_verify_onchain.sh --submit

# With your own account (paymaster still sponsors gas)
STARKNET_PRIVATE_KEY=0x... STARKNET_ACCOUNT_ADDRESS=0x... ./04_verify_onchain.sh --submit --paymaster

# Legacy mode (you pay gas in STRK, uses sncast)
STARKNET_PRIVATE_KEY=0x_your_key ./04_verify_onchain.sh --submit --no-paymaster
```

If your proof artifact has `submission_ready: false` (for example with
aggregated RLC weight binding enabled), `04_verify_onchain.sh` will print the
exact soundness-gate reason. In dry-run it exits cleanly; in submit mode it
fails fast before any transaction is sent.

For `verify_model_gkr_v2`, `verify_model_gkr_v3`, and `verify_model_gkr_v4` artifacts, the submit pipeline validates
that `weight_binding_mode` matches the artifact mode:
- `Sequential` -> `0`
- `BatchedSubchannelV1` -> `1`
- `AggregatedTrustlessV2` -> `2` (v3 only)
- `AggregatedOpeningsV4Experimental` -> `3` (v4 only)
- `AggregatedOracleSumcheck` -> `4` (v4 default)
before sending TX.
For v3/v4, it enforces:
- `weight_binding_data=[]` in modes `0/1`
- non-empty `weight_binding_data` in mode `2`
 - non-empty `weight_binding_data` in mode `3`
 - non-empty `weight_binding_data` in mode `4`
It also enforces fail-fast calldata bounds:
- `OBELYSK_MAX_GKR_CALLDATA_FELTS` (default `300000`)
- `OBELYSK_MAX_GKR_MODE4_CALLDATA_FELTS` (default `120000`)
- `OBELYSK_MIN_GKR_MODE4_CALLDATA_FELTS` (default `1000`)
If the target contract does not expose your requested entrypoint
(`verify_model_gkr_v2`, `verify_model_gkr_v3`, or `verify_model_gkr_v4`), submit with v1
(`--starknet-ready --legacy-gkr-v1`) or deploy the upgraded verifier first.
The paymaster submit path now preflights contract ABI support and fails fast
before transaction submission when the requested entrypoint is missing.

The script will:
1. Auto-detect submission mode (paymaster on Sepolia when no key, sncast otherwise)
2. Auto-install Node.js + starknet.js if needed (for paymaster mode)
3. Submit the proof transaction (gasless via AVNU paymaster, or via sncast)
4. Wait for confirmation (~30 seconds)
5. Check `is_verified()` on the contract
6. Print the explorer link

---

## Quick Reference

### One-Command Bootstrap (fresh machine)

```bash
# Single prompt + on-chain (zero-config)
curl -fsSL https://raw.githubusercontent.com/Bitsage-Network/obelyzk.rs/main/scripts/pipeline/bootstrap.sh | bash -s -- --preset qwen3-14b --gpu --submit

# Multi-turn conversation + on-chain
curl -fsSL https://raw.githubusercontent.com/Bitsage-Network/obelyzk.rs/main/scripts/pipeline/bootstrap.sh | bash -s -- --preset qwen3-14b --gpu --submit --conversation-topic "quantum computing"

# 5-turn conversation
curl -fsSL https://raw.githubusercontent.com/Bitsage-Network/obelyzk.rs/main/scripts/pipeline/bootstrap.sh | bash -s -- --preset qwen3-14b --gpu --submit --conversation-topic "machine learning" --conversation-turns 5

# Dry run (no on-chain)
curl -fsSL https://raw.githubusercontent.com/Bitsage-Network/obelyzk.rs/main/scripts/pipeline/bootstrap.sh | bash -s -- --preset phi3-mini --gpu --dry-run
```

### run_e2e.sh Examples (already cloned)

```bash
# Test everything locally (no on-chain, smallest model)
./run_e2e.sh --preset phi3-mini --gpu --dry-run

# Full pipeline with zero-config on-chain verification (Sepolia)
# Automatically uses a complex mathematical prompt for real tokenized inference
./run_e2e.sh --preset qwen3-14b --gpu --submit

# Multi-turn conversation — real text responses with ZK proofs
./run_e2e.sh --preset qwen3-14b --gpu --submit --conversation-topic "quantum computing"
./run_e2e.sh --preset qwen3-14b --gpu --submit --conversation-topic "cryptography" --conversation-turns 5

# Custom prompt — prove real inference on your own text
./run_e2e.sh --preset qwen3-14b --gpu --submit \
  --prompt "Prove that for any prime p > 2, the Legendre symbol (a/p) satisfies Euler's criterion: a^((p-1)/2) ≡ (a/p) mod p"

# Random input (disable default prompt)
./run_e2e.sh --preset qwen3-14b --gpu --submit --no-prompt

# Legacy v1 sequential openings path
./run_e2e.sh --preset qwen3-14b --gpu --submit --legacy-gkr-v1

# With your own account (legacy sncast, you pay gas)
STARKNET_PRIVATE_KEY=0x... ./run_e2e.sh --preset qwen3-14b --gpu --submit --no-paymaster

# Gated model with HF auth + on-chain
HF_TOKEN=hf_xxx ./run_e2e.sh --preset llama3-8b --gpu --submit

# Chat with the model before proving
./run_e2e.sh --preset phi3-mini --gpu --chat --dry-run

# Resume from a failed step
./run_e2e.sh --preset qwen3-14b --resume-from prove --gpu --submit

# Resume from capture step
./run_e2e.sh --preset qwen3-14b --resume-from capture --gpu --submit

# Skip audit (audit is on by default)
./run_e2e.sh --preset phi3-mini --gpu --dry-run --no-audit

# Skip setup (machine already configured)
./run_e2e.sh --preset qwen3-14b --skip-setup --gpu --submit
```

### Environment Variables

| Variable | What it does |
|----------|--------------|
| `STARKNET_PRIVATE_KEY` | Your Starknet wallet key (optional on Sepolia — paymaster is default) |
| `STARKNET_ACCOUNT_ADDRESS` | Account address (when using own key with paymaster) |
| `HF_TOKEN` | HuggingFace token (for gated models) |
| `OBELYSK_DEPLOYER_KEY` | Deployer key for factory account creation |
| `OBELYSK_DEPLOYER_ADDRESS` | Deployer address for factory account creation |
| `DRY_RUN=1` | Print commands without running them |
| `OBELYSK_DEBUG=1` | Show verbose debug output |

### Where Things Are Saved

```
~/.obelysk/
  models/phi3-mini/       <- Downloaded model files
  logs/phi3-mini/         <- Captured inference logs (from 02b)
  proofs/phi3-mini_.../   <- Generated proofs
  llama.cpp/              <- Built llama.cpp (for inference testing)
  gpu_config.env          <- Detected GPU info
```

---

## GPU Compatibility

| GPU | VRAM | Works? |
|-----|------|--------|
| RTX 4090 | 24GB | Yes (small models or few layers) |
| A100 | 40-80GB | Yes |
| H100 | 80GB | Yes |
| H200 | 141GB | Yes |
| B200 | 192GB | Yes |
| B300 | 288GB | Yes (can prove 70B+ models) |

---

## Troubleshooting

**"nvidia-smi not found"**
```bash
./00_setup_gpu.sh --install-drivers
# Reboot if needed: sudo reboot
```

**"CUDA not found"**
```bash
./00_setup_gpu.sh --install-drivers
```

**"Permission denied" on model download**
The model is gated. Get a HuggingFace token and set `HF_TOKEN`.

**"Not enough disk space"**
You need model_size x 1.5 free. For Qwen3-14B, that's ~50GB.

**Proof fails on-chain but passes locally**
```bash
# Check TX status
sncast tx-status 0xYOUR_TX_HASH
```

**Want to start over?**
```bash
rm -rf ~/.obelysk
```
