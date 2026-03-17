# ObelyZK Protocol Roadmap

**Version**: 1.0 | **Date**: March 16, 2026 | **Author**: Bitsage Network

> Strategy: Deep research per phase, implement, test end-to-end on H100 GPU, verify all security properties, then advance to next phase.

---

## Current Baseline (March 16, 2026)

| Metric | Value |
|--------|-------|
| Model | Qwen3-14B (9.2B params, 40 layers, 160 MatMuls) |
| Prove time (cached) | **103s** on H100 NVL |
| Audit (3 inferences) | **5m 11s** |
| Security tests | 41/41 pass (18 tamper + 23 integration gates) |
| On-chain | Starknet Sepolia, 18-TX streaming, v34 Cairo verifier |
| Proof system | GKR sumcheck + Poseidon2-M31 commitments (no FRI) |
| Field | M31 -> CM31 -> QM31 (124-bit algebraic security) |

---

## Phase 1: Close Soundness Gaps

**Goal**: Every arithmetic operation in a transformer forward pass is provably verified. No unverified operations remain.

**Success criteria**: A malicious prover cannot fabricate any intermediate value without detection.

### 1A. LayerNorm/RMSNorm Mean & Variance Verification

**Status**: CRITICAL GAP — prover can claim arbitrary mean (mu) and variance (sigma^2).

**Current behavior**: LayerNorm reduction accepts mean and variance as trace inputs without constraining them to equal the actual statistics of the input vector.

**Attack**: Malicious prover computes correct MatMul outputs but substitutes arbitrary normalization parameters, producing different outputs while passing verification.

**Protocol fix**:

1. **Mean verification** via inner-product sumcheck:
   - Claim: `mean = (1/n) * sum(x_i)` for input vector x of length n
   - Reduce to: `n * mean = sum(x_i)`
   - Express as inner product: `<x, ones> = n * mean`
   - Standard inner-product sumcheck (log(n) rounds)
   - Cost: ~log(5120) = 13 sumcheck rounds per norm layer

2. **Variance verification** via degree-3 eq-sumcheck:
   - Claim: `variance = (1/n) * sum((x_i - mean)^2)`
   - Expand: `n * variance = sum(x_i^2) - 2*mean*sum(x_i) + n*mean^2`
   - Since `sum(x_i) = n*mean` (from step 1): `n * variance = sum(x_i^2) - n*mean^2`
   - Reduce to: prove `sum(x_i^2)` via sumcheck over MLE of x^2
   - Cost: ~13 sumcheck rounds + 1 MLE evaluation

3. **Integration**:
   - Add `MeanVarianceProof` to `LayerProof::LayerNorm` and `LayerProof::RMSNorm`
   - Verifier replays sumcheck and checks consistency
   - Mix proof into Fiat-Shamir channel between norm reduction and next layer

**Files to modify**:
- `src/gkr/prover.rs`: Add mean/variance sumcheck after norm reduction
- `src/gkr/verifier.rs`: Verify mean/variance proof
- `src/gkr/types.rs`: Add `MeanVarianceProof` type
- `src/components/layernorm.rs`: Compute and export witness
- `src/components/rmsnorm.rs`: Same (variance only, no mean for RMSNorm)

**Estimated calldata overhead**: ~200 felts per norm layer (13 rounds * 3 coefficients + evaluations). For 81 RMSNorm + 0 LayerNorm in Qwen3-14B: ~16,200 extra felts (~19% increase).

**Estimated proving overhead**: ~1-2s total (81 norm layers * ~15ms each).

**Test plan**:
- Unit test: prove mean/variance for known vector, verify
- Tamper test: corrupt mean, verify rejection
- Tamper test: corrupt variance, verify rejection
- Integration: full 40-layer proof with mean/variance constraints
- H100 benchmark: measure overhead vs baseline

---

### 1B. Softmax Sum Constraint

**Status**: CRITICAL GAP — verifier accepts any `sum_exp` value without checking it equals the actual sum of exponentials.

**Current behavior**: `SoftmaxNormEval` constraint checks `weights[i] * sum_exp == exp_values[i]` but `sum_exp` is provided by the prover as an input, not derived from `exp_values`.

**Attack**: Malicious prover claims `sum_exp = 1`, making `weights[i] = exp_values[i]` (unnormalized softmax). Verifier accepts.

**Additional bug**: If `sum_exp == 0 (mod P)`, the forward pass returns unnormalized exp values (attention.rs line 856-863). This is a correctness bug independent of the soundness gap.

**Protocol fix**:

1. **Sum accumulator column**: Add a running-sum interaction column to the SoftmaxNorm STARK component that proves `sum_exp = sum(exp_values[0..seq_len])`.

2. **Implementation**: In `SoftmaxNormEval`, add constraint:
   ```
   accumulator[0] = exp_values[0]
   accumulator[i] = accumulator[i-1] + exp_values[i]
   accumulator[seq_len-1] = sum_exp
   ```

3. **Guard clause**: Fix the sum=0 edge case — reject or handle gracefully.

**Files to modify**:
- `src/components/attention.rs`: Add accumulator column, fix sum=0 bug
- `src/gkr/verifier.rs`: Verify accumulator constraint
- `src/aggregation.rs`: Wire accumulator into trace building

**Estimated overhead**: ~50 felts per attention layer, ~1s total proving.

**Test plan**:
- Unit test: verify correct softmax sum
- Tamper test: wrong sum_exp, verify rejection
- Edge test: all-zero input (sum_exp = seq_len), verify behavior
- Edge test: sum_exp = 0 mod P, verify graceful handling
- H100 benchmark: full model with softmax sum constraint

---

### 1C. RoPE Position Encoding Arithmetization

**Status**: CRITICAL GAP — RoPE is compiled as `LayerType::Identity`, meaning the rotation is completely unverified.

**Current behavior**: `GraphOp::RoPE => LayerType::Identity` (circuit.rs:271). The GKR claim propagates unchanged — no rotation constraint is checked.

**Attack**: Malicious prover can skip RoPE entirely or apply wrong rotations, producing attention patterns that don't correspond to the model's actual position encoding.

**Why this is hard**: RoPE uses trigonometric functions (`cos`, `sin`) which are computed via f64 in the current implementation. Proving cos/sin in a finite field requires either:
- (a) Precomputed lookup tables (LogUp, like activations)
- (b) Taylor series approximation (high-degree polynomial constraint)
- (c) Cordic-style iterative computation (complex circuit)

**Protocol fix (recommended: LogUp table approach)**:

1. **Precompute angle table**: For each position `pos` in [0, max_seq_len] and each dimension pair `j` in [0, d/2]:
   - `theta_j = base^(-2j/d)` (computed in integer fixed-point)
   - `cos_table[pos][j] = fixed_point_cos(pos * theta_j)`
   - `sin_table[pos][j] = fixed_point_sin(pos * theta_j)`

2. **Commit table**: Poseidon Merkle root of the angle table (registered on-chain like weight commitments).

3. **Prove rotation**: For each (pos, dim_pair), prove via LogUp:
   - `(pos, j, cos_val, sin_val)` is in the committed table
   - Output constraint: `out[2j] = in[2j]*cos - in[2j+1]*sin`, `out[2j+1] = in[2j]*sin + in[2j+1]*cos`

4. **Integer-only computation**: Replace f64 `powf/cos/sin` with fixed-point integer arithmetic using Chebyshev polynomials or precomputed tables. This eliminates platform divergence.

**Files to modify**:
- `src/gkr/circuit.rs`: Change `GraphOp::RoPE => LayerType::RoPE { .. }` (new layer type)
- `src/gkr/prover.rs`: Add `reduce_rope_layer()` with LogUp proof
- `src/gkr/verifier.rs`: Add RoPE verification
- `src/components/rope.rs`: Integer-only angle computation + table generation
- `src/aggregation.rs`: Wire RoPE into unified STARK for LogUp

**Estimated overhead**: ~500 felts per attention layer (table lookup proof). For 40 layers: ~20,000 extra felts.

**Estimated proving overhead**: ~3-5s total (table commitment + LogUp for 40 layers).

**Test plan**:
- Unit test: prove rotation for known angles
- Tamper test: wrong rotation angle, verify rejection
- Tamper test: wrong position offset, verify rejection
- Consistency test: integer vs f64 computation match
- H100 benchmark: full model with RoPE constraints

---

### 1D. Causal Attention Mask Verification

**Status**: CRITICAL GAP — prover applies mask in forward pass but verifier doesn't check which positions are masked.

**Current behavior**: Causal mask sets `scores[i][j] = P-2` for `j > i + cache_offset`. This happens in the prover's forward pass but the softmax STARK component doesn't constrain it.

**Attack**: Malicious prover masks wrong positions (e.g., allows future tokens), breaking autoregressive property.

**Protocol fix**:

1. **Mask constraint in SoftmaxExp**: For each score entry, the constraint checks:
   ```
   if col_idx > row_idx + position_offset:
       score must equal MASK_VALUE (P-2)
   ```

2. **Implementation**: Add a preprocessed column encoding the mask pattern. The STARK constraint verifies that masked positions have the sentinel value.

3. **Position offset binding**: Link `position_offset` to the KV-cache commitment chain, ensuring consistency across decode steps.

**Files to modify**:
- `src/components/attention.rs`: Add mask constraint column
- `src/aggregation.rs`: Wire mask into trace

**Estimated overhead**: Minimal — mask is a preprocessed column (no prover work).

**Test plan**:
- Unit test: correct mask accepted
- Tamper test: future token unmasked, verify rejection
- Decode test: mask consistent with KV-cache position

---

### 1E. Eliminate f64 Platform Divergence

**Status**: CRITICAL — RoPE angles, activation tables, quantization all use f64. Different platforms may produce different M31 values.

**Current behavior**: `base.powf(-2.0 * j / d)`, `angle.cos()`, `angle.sin()`, GELU tanh approximation all use hardware f64.

**Protocol fix**:

1. **Integer-only RoPE**: Chebyshev polynomial cos/sin over fixed-point M31 values
2. **Integer-only activation tables**: Precompute GELU/Sigmoid/Softmax tables using only M31 arithmetic (polynomial approximation or exhaustive enumeration for 16-bit range)
3. **Deterministic quantization**: Replace f32 intermediate with M31-native rounding

**Test plan**:
- Cross-platform test: compute tables on x86, ARM, GPU — verify identical M31 outputs
- Roundtrip test: encode -> decode matches for all valid inputs

---

## Phase 2: Performance — Sub-30s Proving

**Goal**: Full 40-layer Qwen3-14B proof in under 30 seconds on H100 NVL.

### 2A. Lower GPU Dispatch Threshold

**Current**: GPU sumcheck only activates for matrices with k >= 16384 elements.
**Issue**: Qwen3-14B Q/K/V projections are 5120x5120 (k=5120 < 16384), so they fall to CPU.
**Fix**: Change threshold to 4096 (or even 2048).
**Expected savings**: ~25s (forward pass 38s->15s, GKR walk 47s->30s).

**Files**: `src/gpu_sumcheck.rs` line 14 (MLE_THRESHOLD constant)

**Test plan**: Benchmark 1-layer and 40-layer before/after on H100.

---

### 2B. GPU-Accelerated Unified STARK

**Current**: Unified STARK uses SimdBackend (CPU) because GpuBackend hits `ConstraintsNotSatisfied` with preprocessed columns.
**Fix**: Debug and fix GpuBackend preprocessed column handling.
**Expected savings**: ~3s (5s -> 2s).

**Files**: `src/aggregation.rs` (STARK backend dispatch)

---

### 2C. Fused Forward + Activation GPU Kernels

**Current**: Forward pass downloads matmul output to CPU, applies activation, uploads back.
**Fix**: CUDA kernel that applies activation in-place on GPU after matmul.
**Expected savings**: ~5s (eliminate CPU<->GPU round-trips).

**Files**: `src/gpu_sumcheck.rs` (new activation kernel), `src/aggregation.rs` (dispatch)

---

### 2D. GPU Memory Pooling

**Current**: Each matmul allocates fresh GPU buffers (`device.alloc_zeros()`).
**Fix**: Pre-allocate buffer pool, reuse across layers.
**Expected savings**: ~2s (eliminate allocation overhead for 160 matmuls).

---

### 2E. Binary Calldata Serialization

**Current**: JSON hex encoding (`format!("0x{:x}", f)` per element).
**Fix**: Compact binary format for storage/transmission. JSON kept for debug.
**Expected savings**: ~1.5s, 7MB -> 3.6MB proof size.

---

### Phase 2 Target

| Metric | Before | After |
|--------|--------|-------|
| Forward pass | 38s | ~12s |
| GKR walk | 47s | ~25s |
| Unified STARK | 5s | ~2s |
| Serialization | 2s | ~0.5s |
| **Total** | **103s** | **~30s** |

---

## Phase 3: Multi-Model Support & Benchmarking

**Goal**: Prove the protocol works across diverse architectures, not just Qwen3-14B.

### 3A. Model Support Matrix

| Model | Params | Architecture | New Components Needed | Priority |
|-------|--------|-------------|----------------------|----------|
| **Qwen3-14B** | 14B | Dense transformer, GQA, RMSNorm | None (baseline) | DONE |
| **MiniMax-01** | 456B | MoE (32 experts, top-2 routing) | Router verification, expert selection proof | HIGH |
| **Kimi (Moonshot)** | ~200B | Dense, 200K context | Long-context KV proving, chunked attention | HIGH |
| **YOLOv8** | 3-68M | CNN + detection head | Conv2D, BatchNorm, anchor box | MEDIUM |
| **Llama-3-8B** | 8B | Dense transformer, GQA, RMSNorm | None (same arch as Qwen) | HIGH |
| **Llama-3-70B** | 70B | Dense, GQA | Multi-GPU proving, memory optimization | HIGH |
| **Mistral-7B** | 7B | Sliding window attention | Window mask constraint | MEDIUM |
| **Mixtral-8x7B** | 47B | MoE (8 experts, top-2) | Router verification | HIGH |
| **Phi-3 Mini** | 3.8B | Dense, long context | Already partially supported | LOW |
| **GPT-2** | 124M-1.5B | Dense, LayerNorm (not RMSNorm) | LayerNorm gamma/beta | LOW |
| **DeepSeek-V3** | 671B | MoE (256 experts) | Large-scale MoE routing | FUTURE |

### 3B. MoE (Mixture-of-Experts) Verification

MoE models (MiniMax, Mixtral, DeepSeek) require two new verified components:

1. **Router verification**: Prove the gating network's top-k selection is correct
   - Router is a small MatMul (hidden_dim -> num_experts) — already verifiable
   - Top-k selection: prove the k largest values were correctly identified
   - New component: `TopKEval` STARK with sorting network constraint

2. **Expert selection proof**: Prove only the selected experts were activated
   - Conditional execution: if expert_i selected, prove MatMul; if not, prove zero contribution
   - Challenge: variable computation graph per token (experts differ)
   - Solution: commit to routing decisions, prove each selected expert independently

### 3C. CNN Support (YOLOv8)

Conv2D can be decomposed into MatMul via im2col:
- `Conv2D(input, kernel) = MatMul(im2col(input), reshape(kernel))`
- im2col is a deterministic reorganization (index mapping) — provable via permutation argument
- BatchNorm: same structure as LayerNorm (mean + variance + affine transform)
- Detection head: argmax/NMS are post-processing, not part of the arithmetic trace

### 3D. Competitive Benchmarking

| Competitor | Claimed | Architecture | Our Advantage |
|-----------|---------|-------------|---------------|
| **EZKL** | ~1M params, seconds | Halo2/KZG | We handle 14B+ params, on-chain verifier |
| **zkLLM** | 13B params, 1-15 min | Custom commitment | We're faster (103s), deployed on-chain |
| **Giza** | ~10M params, minutes | Cairo-STARK | We're 1000x larger model scale |
| **Expander** | GKR framework | GKR (no deployment) | We have deployed Cairo verifier |
| **DeepProve** | GPT-2 scale | GKR | We handle 100x larger models |

**Benchmark protocol**:
1. Standardize on common models (GPT-2, Llama-3-8B) for apples-to-apples comparison
2. Measure: prove time, verify time, proof size, calldata size, security level
3. Publish reproducible benchmarks with exact hardware specs
4. Open-source benchmark suite for community verification

---

## Phase 4: Protocol Innovation

### 4A. Recursive Proof Composition

**Goal**: Compress 112K felt calldata into ~500 felts via recursive STARK.

**Approach**: Prove the GKR verifier execution in a STARK circuit (the verifier becomes the witness). The recursive proof attests "I verified the GKR proof and it passed." On-chain, verify only the recursive proof.

**Impact**: Constant-size on-chain verification regardless of model size. A 14B model and a 400B model have the same verification cost.

### 4B. Streaming Proof Aggregation

**Goal**: Aggregate multiple inference proofs into a single on-chain proof.

**Approach**: Batch N inference proofs into one recursive proof. On-chain verifier checks one proof for N inferences.

**Use case**: Audit pipeline proves 100 inferences, submits 1 aggregated proof. Cost: ~0.25 STRK total instead of 100 * 0.25 = 25 STRK.

### 4C. Verifiable Fine-Tuning

**Goal**: Prove that fine-tuning was performed correctly (DP-SGD, LoRA, etc.)

**Approach**: Extend GKR to backward pass gradients. Prove:
- Loss function evaluation
- Gradient computation (chain rule through verified forward pass)
- Weight update rule (SGD/Adam step)
- Differential privacy noise injection (if required)

### 4D. Cross-Model Pipeline Verification

**Goal**: Prove that a pipeline (model A -> post-process -> model B) executed correctly.

**Approach**: Chain proof commitments across models. Proof A's output commitment becomes Proof B's input commitment. The on-chain verifier checks both proofs and the linking commitment.

---

## Phase 5: Production Infrastructure

### 5A. Inference Engine Integration

**Goal**: Drop-in plugin for vLLM, TGI, Ollama that captures M31 intermediates.

**Approach**:
- Hook into the inference engine's forward pass at the matmul/norm/activation boundaries
- Capture intermediates as M31 matrices (quantize from f16/bf16)
- Write to append-only inference log
- Proving runs asynchronously from serving

### 5B. Prover Fleet Orchestration

**Goal**: Horizontal scaling across GPU fleet.

**Approach**:
- Queue-based: inference log entries → prover job queue → GPU workers
- Worker pools: dedicated H100s for proving (separate from serving)
- Priority scheduling: high-value inferences first

### 5C. Verification API

**Goal**: Public REST API for proof verification.

**Approach**:
- `GET /verify/{proof_hash}` → verified/not-verified + model details
- `GET /model/{model_id}/proofs` → list of verified proofs
- `POST /challenge/{inference_id}` → trigger on-demand proof generation

---

## Execution Timeline

| Phase | Duration | Deliverable | Test Gate |
|-------|----------|-------------|-----------|
| **1A** LayerNorm mean/variance | 1-2 weeks | Inner-product sumcheck for mean + variance | Full 40L proof on H100, tamper tests pass |
| **1B** Softmax sum | 1 week | Sum accumulator constraint | Attention proof on H100, tamper tests pass |
| **1C** RoPE arithmetization | 2-3 weeks | LogUp table + rotation constraint | Full 40L with position encoding verified |
| **1D** Causal mask | 1 week | Mask constraint in STARK | Attention with correct/incorrect masks tested |
| **1E** f64 elimination | 2 weeks | Integer-only tables | Cross-platform determinism test |
| **2A** GPU threshold | 1 day | Constant change + benchmark | 40L in <80s on H100 |
| **2B** GPU unified STARK | 1-2 weeks | Fix GpuBackend | 40L in <75s on H100 |
| **2C** Fused kernels | 2 weeks | New CUDA kernels | 40L in <50s on H100 |
| **2D-2E** Memory + serialization | 1 week | Pool + binary format | 40L in <35s on H100 |
| **3A-3D** Multi-model + benchmarks | 4-6 weeks | 5+ models verified, benchmark suite | Published comparison table |
| **4A** Recursive composition | 4-6 weeks | Constant-size on-chain proof | Single TX verification on Starknet |

---

## Competitive Position After Roadmap

| Capability | EZKL | zkLLM | Giza | Expander | **ObelyZK (Target)** |
|-----------|------|-------|------|----------|---------------------|
| Max params | 1M | 13B | 10M | ? | **400B+ (MoE)** |
| Prove time | seconds | 1-15 min | minutes | ? | **<30s (14B dense)** |
| Full semantics | Yes | Yes | Yes | ? | **Yes (all ops verified)** |
| On-chain verifier | Solidity | None | Cairo | None | **Cairo (Starknet)** |
| Proof system | Halo2/KZG | Custom | STARK | GKR | **GKR + STARK (no FRI)** |
| MoE support | No | No | No | ? | **Yes (router + expert)** |
| CNN support | Yes | No | No | ? | **Yes (im2col)** |
| Recursive proofs | No | No | No | ? | **Yes (constant-size)** |
| Multi-inference audit | No | No | No | No | **Yes (batch proving)** |
| KV-cache chain | No | No | No | No | **Yes (incremental Merkle)** |
| Deployed | Ethereum | No | Starknet | No | **Starknet** |

---

## Development Methodology

For each phase item:

1. **Research**: Deep dive into the protocol extension, write formal specification
2. **Implement**: Code the prover and verifier changes
3. **Unit test**: Local tests with small matrices (fast iteration)
4. **Security test**: Tamper detection tests (corrupt each witness element)
5. **Integration test**: Full pipeline with existing test suite (41+ tests must pass)
6. **H100 GPU test**: End-to-end on real hardware with real model weights
7. **Benchmark**: Measure overhead vs baseline, report in this document
8. **Paper update**: Update obelyzk-paper.tex with new results
9. **Push**: Commit, push, verify CI passes

No phase item is considered complete until it passes the H100 GPU end-to-end test with the full 40-layer Qwen3-14B model.
