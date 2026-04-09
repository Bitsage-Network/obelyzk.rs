<div align="center">

<br/>

```
                    ██████╗ ██████╗ ███████╗██╗  ██╗   ██╗███████╗██╗  ██╗
                   ██╔═══██╗██╔══██╗██╔════╝██║  ╚██╗ ██╔╝╚══███╔╝██║ ██╔╝
                   ██║   ██║██████╔╝█████╗  ██║   ╚████╔╝   ███╔╝ █████╔╝
                   ██║   ██║██╔══██╗██╔══╝  ██║    ╚██╔╝   ███╔╝  ██╔═██╗
                   ╚██████╔╝██████╔╝███████╗███████╗██║   ███████╗██║  ██╗
                    ╚═════╝ ╚═════╝ ╚══════╝╚══════╝╚═╝   ╚══════╝╚═╝  ╚═╝
                                          ·  r s  ·
```

<br/>

### Verifiable AI Engine Written in Rust

**Provable inference for every model.**

<br/>

[![Rust](https://img.shields.io/badge/rust-nightly--2025--07--14-orange?style=for-the-badge)](https://rust-lang.org)
[![CUDA](https://img.shields.io/badge/CUDA-12%2B-76B900?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Starknet](https://img.shields.io/badge/Starknet-Sepolia-29296E?style=for-the-badge)](https://starknet.io)
[![License](https://img.shields.io/badge/Apache--2.0-grey?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-950_passing-brightgreen?style=for-the-badge)]()

<br/>

[Quick Start](#quick-start) · [Architecture](#architecture) · [Performance](#performance) · [Models](#supported-models) · [On-Chain](#on-chain-verification) · [SDKs](#sdks)

<br/>

---

</div>

<br/>

> **obelyzk.rs** is a purpose-built execution environment where every AI computation produces a cryptographic proof — verified on Starknet in a single transaction. Open-weight models get full ZK proofs. Closed-source APIs get TLS attestation. Every model, every provider, every response — verifiable.

<br/>

## Quick Start

```bash
# Chat with Claude — every response TLS-attested
ANTHROPIC_API_KEY=sk-ant-... obelyzk chat --model claude-sonnet

# Chat with a local model — every response ZK-proved
OBELYSK_MODEL_DIR=./models/smollm2-135m obelyzk chat

# Serve an OpenAI-compatible API with built-in proving
OBELYSK_MODEL_DIR=./models/qwen3-14b obelyzk serve --port 8080

# Benchmark proving throughput
OBELYSK_MODEL_DIR=./models/qwen3-14b obelyzk bench --tokens 10000
```

```bash
obelyzk chat         # Interactive verified chat with any model
obelyzk serve        # OpenAI-compatible API server with proving
obelyzk bench        # Throughput benchmark (80-322 tok/s on H100)
obelyzk prove        # Prove a model inference
obelyzk verify       # Verify a proof offline
```

<br/>

## Architecture

```
obelyzk.rs/
│
├── engine/              The ObelyZK proving engine (crate: obelyzk)
│   └── src/
│       ├── gkr/                  GKR sumcheck prover · CPU + 19 CUDA kernels
│       ├── vm/                   VM runtime · trace, executor, queue, providers
│       ├── providers/            Local (ZK) · OpenAI · Anthropic (TLS attestation)
│       ├── components/           MatMul · Attention · Norm · Embedding · RoPE · TopK
│       ├── compiler/             HuggingFace model loader · graph compiler
│       ├── recursive/            Recursive STARK compression (46K → 942 felts)
│       └── aggregation.rs        Full proving pipeline · batched throughput
│
├── stwo-gpu/            STWO Circle STARK prover + our GPU backend
│                        CUDA kernel dispatch · GPU FRI · GPU quotient eval
│
├── elo-cairo-verifier/  Recursive STARK verifier on Starknet (Cairo)
├── stark-cairo/         STWO STARK verifier in Cairo
├── verifier/            ML proof verifier
├── proof-stream/        Real-time proof visualization (WebSocket)
│
├── sdk/
│   ├── python/          pip install obelyzk
│   ├── typescript/      npm install @obelyzk/sdk
│   └── cli/             npm install -g @obelyzk/cli
│
└── scripts/             Deployment, benchmarking, on-chain submission
```

<details>
<summary><b>STWO Foundation</b></summary>
<br/>

Built on **[STWO](https://github.com/starkware-libs/stwo)** by StarkWare — the Circle STARK prover. Our fork (`stwo-gpu/`) adds the complete GPU proving backend:

- CUDA kernel dispatch + CudaStream graph execution
- GPU FRI layer folding
- GPU quotient evaluation
- Felt252Dict removal for Starknet Sierra 1.7 compatibility
- Preprocessed column deduplication fixes

The GKR protocol, ML inference pipeline, VM runtime, TLS attestation, and on-chain verification are original work by [Bitsage Network](https://bitsage.network).

</details>

<br/>

## Performance

<div align="center">

| Model | Hardware | Batch | Throughput | Proof Type |
|:------|:---------|------:|----------:|:-----------|
| SmolLM2-135M | A10G | 1 | 0.05 tok/s | GKR + STARK |
| SmolLM2-135M | A10G | 8 | 0.23 tok/s | GKR + STARK |
| Qwen3-14B | 1×H100 | 10K | **80 tok/s** | GKR + STARK |
| Qwen3-14B | 1×H100 (opt.) | 10K | **322 tok/s** | GKR + STARK |
| Qwen3-14B | 8×H100 | 10K | **~1,600 tok/s** | GKR + STARK |
| Claude / GPT / Grok | Any | Stream | **Instant** | TLS attestation |

</div>

> **Recursive STARK compression**: 46,148 GKR felts → 942 felts. Verified in one Starknet transaction.

<br/>

## On-Chain Verification

Every proof is verified trustlessly on Starknet. No optimistic assumptions. No fraud proofs. No committee.

<div align="center">

| Contract | Address |
|:---------|:--------|
| **Recursive Verifier** | [`0x1c208a...0c7`](https://sepolia.starkscan.co/contract/0x1c208a5fe731c0d03b098b524f274c537587ea1d43d903838cc4a2bf90c40c7) |
| **Verified TX** | [`0x276c6a...ddc`](https://sepolia.starkscan.co/tx/0x276c6a448829c0f3975080914a89c2a9611fc41912aff1fddfe29d8f3364ddc) |
| **Network** | Starknet Sepolia |
| **Verification** | OODS + Merkle + FRI + PoW |

</div>

<br/>

## Supported Models

> Every model. Every provider. Every trust level.

### Open-Weight Models — Full ZK Proof

GKR sumcheck over every operation. Recursive STARK compression. Weight binding via Poseidon Merkle roots. Verified on-chain.

<div align="center">

| Architecture | Models | Status |
|:------------|:-------|:------:|
| **LLaMA** | Llama-3.x, SmolLM2, CodeLlama | ✅ Proven |
| **Qwen** | Qwen2, Qwen3-14B | ✅ Proven |
| **Phi** | Phi-3, Phi-4 | ✅ Proven |
| **Mistral** | Mistral-7B, Mixtral-8x7B (MoE) | ✅ Proven |
| **Yi** | Yi-1.5-6B | ✅ Proven |
| **Gemma** | Gemma-2 | ✅ Auto-detect |
| **DeepSeek** | DeepSeek-V2, DeepSeek-R1 | ✅ Auto-detect |
| **GLM** | ChatGLM, GLM-4 | ✅ Auto-detect |
| **MiniMax** | MiniMax-01, MiniMax-Text | ✅ Auto-detect |
| **Falcon** | Falcon-7B, Falcon-40B | ✅ Auto-detect |
| **MPT** | MPT-7B, MPT-30B | ✅ Auto-detect |
| **RWKV** | RWKV-6 | ✅ Auto-detect |
| **Any HuggingFace** | SafeTensors format | ✅ Auto-detect |

</div>

### Closed-Source APIs — TLS Attestation

Cryptographic proof that the API call happened. Certificate-verified commitment over request + response. Tamper-evident.

<div align="center">

| Provider | Models | Trust |
|:---------|:-------|:-----:|
| **Anthropic** | Claude Opus, Sonnet, Haiku | 🔒 TLS |
| **OpenAI** | GPT-4o, o1, o3, GPT-4 | 🔒 TLS |
| **Google** | Gemini Pro, Ultra, Flash | 🔒 TLS |
| **xAI** | Grok-2, Grok-3 | 🔒 TLS |
| **DeepSeek** | DeepSeek-Chat API | 🔒 TLS |
| **MiniMax** | abab-7B-chat | 🔒 TLS |
| **Any OpenAI-compatible** | vLLM, TGI, Ollama, LM Studio | 📋 Commitment |

</div>

<br/>

## SDKs

```bash
pip install obelyzk          # Python
npm install @obelyzk/sdk     # TypeScript
npm install -g @obelyzk/cli  # CLI
```

```python
from obelyzk import ObelyzkClient

client = ObelyzkClient()
result = await client.chat("smollm2-135m", "What is ZKML?")
print(result.predicted_text)   # model response
print(result.proof_id)         # cryptographic proof
print(result.io_commitment)    # Poseidon commitment
```

<br/>

## Building

```bash
# Prerequisites: Rust nightly-2025-07-14, CUDA 12+ (optional)
rustup toolchain install nightly-2025-07-14

# Build the engine
cd engine && cargo build --release --bin obelyzk --features "server,cuda-runtime"

# Run tests (950 passing)
cargo test --lib --features std
```

<br/>

---

<div align="center">

**950 tests · 1,750+ commits · Verifiable AI for every model**

[obelysk.xyz](https://obelysk.xyz) · [bitsage.network](https://bitsage.network) · [Starknet Sepolia](https://sepolia.starkscan.co/contract/0x1c208a5fe731c0d03b098b524f274c537587ea1d43d903838cc4a2bf90c40c7)

Apache-2.0

</div>
