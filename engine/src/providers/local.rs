//! Local inference provider — runs the M31 forward pass with full ZK proof.
//!
//! This is the highest-trust provider: the computation runs inside the VM
//! and every operation is provable via GKR sumcheck + recursive STARK.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use crate::compiler::graph::{ComputationGraph, GraphWeights};
use crate::components::attention::ModelKVCache;
use crate::components::matmul::M31Matrix;
use crate::policy::PolicyConfig;
use crate::providers::types::*;
use crate::vm::trace::ExecutionTrace;

/// Local inference provider — loads HuggingFace models, runs M31 forward pass.
///
/// Full production engine: GPU kernels, STWO backend, weight cache,
/// batched proving, streaming proof pipeline.
pub struct LocalProvider {
    pub model_name: String,
    pub model_dir: PathBuf,
    pub graph: Arc<ComputationGraph>,
    pub weights: Arc<GraphWeights>,
    pub tokenizer: Arc<tokenizers::Tokenizer>,
    pub weight_commitment: String,
    pub hidden_size: usize,
    /// Shared weight commitment cache — avoids recomputing Poseidon Merkle roots.
    pub weight_cache: Option<crate::weight_cache::SharedWeightCache>,
    /// Number of tokens to batch before proving (default: 1).
    pub batch_size: usize,
}

impl LocalProvider {
    /// Load a model from a HuggingFace directory or GGUF file.
    ///
    /// Auto-detects format:
    /// - `.gguf` file → GGUF loader (llama.cpp format)
    /// - Directory with `config.json` → HuggingFace SafeTensors loader
    pub fn load(model_dir: &Path, name: Option<&str>) -> Result<Self, LocalProviderError> {
        let dir_name = model_dir
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "model".into());

        let model_name = name.unwrap_or(&dir_name).to_string();
        let is_gguf = model_dir.extension().map(|e| e == "gguf").unwrap_or(false);

        eprintln!("[local] Loading model from {}{}...",
            model_dir.display(),
            if is_gguf { " (GGUF)" } else { " (SafeTensors)" },
        );

        let hf = if is_gguf {
            // GGUF path — single .gguf file
            crate::compiler::gguf_loader::load_gguf_model(model_dir, None)
                .map_err(|e| LocalProviderError::LoadFailed(format!("GGUF: {e}")))?
        } else {
            // HuggingFace SafeTensors path — directory with config.json + *.safetensors
            crate::compiler::hf_loader::load_hf_model(model_dir, None)
                .map_err(|e| LocalProviderError::LoadFailed(format!("{e}")))?
        };

        // Load tokenizer — for GGUF, check parent directory or same directory
        let tok_path = if is_gguf {
            // GGUF files often don't have tokenizer.json alongside them
            // Check parent dir, or use a bundled tokenizer
            let parent = model_dir.parent().unwrap_or(Path::new("."));
            let tok_in_parent = parent.join("tokenizer.json");
            let tok_alongside = model_dir.with_extension("tokenizer.json");
            if tok_in_parent.is_file() { tok_in_parent }
            else if tok_alongside.is_file() { tok_alongside }
            else { model_dir.with_file_name("tokenizer.json") }
        } else {
            model_dir.join("tokenizer.json")
        };

        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| LocalProviderError::LoadFailed(format!("tokenizer ({}): {e}", tok_path.display())))?;

        let hidden_size = hf.input_shape.1;
        let num_layers = hf.graph.num_layers();
        let num_weights = hf.weights.weights.len();

        // Detect GPU
        let gpu_name = crate::backend::gpu_device_name().unwrap_or_default();
        let gpu_available = crate::backend::gpu_is_available();

        // Initialize weight cache — pre-warms Poseidon Merkle roots on disk
        let model_id = format!("local-{}", model_name);
        let cache_dir = if is_gguf {
            model_dir.parent().unwrap_or(Path::new("."))
        } else {
            model_dir
        };
        let weight_cache = crate::weight_cache::shared_cache_for_model(cache_dir, &model_id);

        // Batch size from env (default: 1 for streaming, higher for throughput)
        let batch_size: usize = std::env::var("OBELYZK_BATCH_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);

        eprintln!("[local] Loaded: {} ({} layers, {} weights, d_model={})", model_name, num_layers, num_weights, hidden_size);
        if gpu_available {
            eprintln!("[local] GPU: {} (CUDA kernels active)", gpu_name);
        } else {
            eprintln!("[local] CPU/SIMD mode (no GPU detected)");
        }
        eprintln!("[local] Weight cache: initialized");
        eprintln!("[local] Batch size: {batch_size} tokens/proof");

        Ok(Self {
            model_name,
            model_dir: if is_gguf { model_dir.parent().unwrap_or(Path::new(".")).to_path_buf() } else { model_dir.to_path_buf() },
            graph: Arc::new(hf.graph),
            weights: Arc::new(hf.weights),
            tokenizer: Arc::new(tokenizer),
            weight_commitment: "deferred".into(),
            hidden_size,
            weight_cache: Some(weight_cache),
            batch_size,
        })
    }

    /// Run inference from a text prompt.
    ///
    /// Returns (output_text, inference_result, execution_trace).
    pub fn infer_text(
        &self,
        prompt: &str,
        kv_cache: Option<&mut ModelKVCache>,
    ) -> Result<(String, InferenceResult, ExecutionTrace), LocalProviderError> {
        let t_start = Instant::now();

        // 1. Tokenize
        let encoding = self.tokenizer.encode(prompt, false)
            .map_err(|e| LocalProviderError::TokenizeFailed(format!("{e}")))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();
        if token_ids.is_empty() {
            return Err(LocalProviderError::EmptyPrompt);
        }
        let last_token_id = *token_ids.last().unwrap();

        // 2. Embed
        let (input_matrix, _vocab_size) = crate::compiler::hf_loader::load_embedding_row(
            &self.model_dir, self.hidden_size, last_token_id,
        ).map_err(|e| LocalProviderError::EmbedFailed(format!("{e}")))?;

        // 3. Execute + prove
        let trace = crate::vm::executor::execute_and_prove(
            &self.graph,
            &input_matrix,
            &self.weights,
            kv_cache,
            Some(&PolicyConfig::standard()),
            token_ids.clone(),
            self.model_name.clone(),
        ).map_err(|e| LocalProviderError::ProveFailed(format!("{e}")))?;

        // 4. Project to logits for predicted text
        let predicted_text = crate::compiler::hf_loader::project_to_logits(
            &self.model_dir, &trace.output,
        )
        .ok()
        .and_then(|(tid, _)| self.tokenizer.decode(&[tid], true).ok())
        .unwrap_or_default();

        let inference_time_ms = t_start.elapsed().as_millis() as u64;

        let io_commitment = trace.io_commitment;
        let result = InferenceResult {
            text: predicted_text.clone(),
            token_ids: token_ids.clone(),
            num_tokens: token_ids.len(),
            model_id: self.model_name.clone(),
            provider: "local".into(),
            trust_model: TrustModel::ZkProof {
                weight_commitment: self.weight_commitment.clone(),
            },
            io_commitment,
            inference_time_ms,
        };

        Ok((predicted_text, result, trace))
    }

    /// Generate multiple tokens autoregressively.
    ///
    /// Phase 1 (prefill): Embeds ALL prompt tokens into a batch matrix,
    /// runs one forward pass to get the full-context output.
    /// Phase 2 (decode): Uses the last position's output to predict next tokens
    /// in a loop, feeding each new token through the forward pass.
    ///
    /// Note: Without KV-cache in the fast path, each decode step only sees
    /// its own embedding (not the full history). The prefill gives the first
    /// prediction good context, but subsequent tokens degrade. This is a
    /// known limitation of the fast path — the full proving path with KV-cache
    /// maintains full context.
    pub fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        mut on_token: impl FnMut(&str, u32, usize),
    ) -> Result<(String, Vec<u32>), LocalProviderError> {
        // Tokenize the full prompt
        let encoding = self.tokenizer.encode(prompt, false)
            .map_err(|e| LocalProviderError::TokenizeFailed(format!("{e}")))?;
        let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
        if prompt_ids.is_empty() {
            return Err(LocalProviderError::EmptyPrompt);
        }

        // EOS tokens
        let eos_ids: Vec<u32> = vec![
            0, 1, 2,
            self.tokenizer.token_to_id("<|endoftext|>").unwrap_or(u32::MAX),
            self.tokenizer.token_to_id("</s>").unwrap_or(u32::MAX),
            self.tokenizer.token_to_id("<|im_end|>").unwrap_or(u32::MAX),
            self.tokenizer.token_to_id("<|end|>").unwrap_or(u32::MAX),
        ];

        // Phase 1: Prefill — embed ALL prompt tokens as a batch
        // This gives the model full context for the first prediction.
        let input_matrix = crate::compiler::hf_loader::load_embedding_batch(
            &self.model_dir, self.hidden_size, &prompt_ids,
        ).map_err(|e| LocalProviderError::EmbedFailed(format!("{e}")))?;

        // Reshape graph for the prompt length
        let prefill_graph = self.graph.with_seq_len(prompt_ids.len());

        // Run prefill forward pass (full context)
        let output = crate::aggregation::execute_forward_pass_fast(
            &prefill_graph, &input_matrix, &self.weights,
        ).map_err(|e| LocalProviderError::ProveFailed(format!("prefill: {e}")))?;

        // Get first prediction from the last position's output row
        let last_row_start = (output.rows - 1) * output.cols;
        let last_row = M31Matrix {
            rows: 1,
            cols: output.cols,
            data: output.data[last_row_start..last_row_start + output.cols].to_vec(),
        };

        let (mut next_id, _) = crate::compiler::hf_loader::project_to_logits(
            &self.model_dir, &last_row,
        ).map_err(|e| LocalProviderError::ProveFailed(format!("logits: {e}")))?;

        let mut generated_ids: Vec<u32> = Vec::new();
        let mut generated_text = String::new();

        // Phase 2: Decode — generate tokens one at a time
        for step in 0..max_tokens {
            if eos_ids.contains(&next_id) {
                break;
            }

            // Decode token to text
            let token_text = self.tokenizer.decode(&[next_id], true)
                .unwrap_or_default();

            generated_ids.push(next_id);
            generated_text.push_str(&token_text);
            on_token(&token_text, next_id, step);

            // Embed the predicted token and forward pass for next prediction
            let (token_input, _) = crate::compiler::hf_loader::load_embedding_row(
                &self.model_dir, self.hidden_size, next_id,
            ).map_err(|e| LocalProviderError::EmbedFailed(format!("{e}")))?;

            let decode_output = crate::aggregation::execute_forward_pass_fast(
                &self.graph, &token_input, &self.weights,
            ).map_err(|e| LocalProviderError::ProveFailed(format!("decode step {step}: {e}")))?;

            let (predicted, _) = crate::compiler::hf_loader::project_to_logits(
                &self.model_dir, &decode_output,
            ).map_err(|e| LocalProviderError::ProveFailed(format!("logits step {step}: {e}")))?;

            next_id = predicted;
        }

        Ok((generated_text, generated_ids))
    }

    /// Fast inference — forward pass ONLY, no proving.
    ///
    /// Returns the predicted text in ~2-8s instead of ~90s.
    /// Call `prove_deferred()` afterward to generate the ZK proof in the background.
    pub fn infer_fast(
        &self,
        prompt: &str,
    ) -> Result<(String, Vec<u32>, M31Matrix), LocalProviderError> {
        // 1. Tokenize
        let encoding = self.tokenizer.encode(prompt, false)
            .map_err(|e| LocalProviderError::TokenizeFailed(format!("{e}")))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();
        if token_ids.is_empty() {
            return Err(LocalProviderError::EmptyPrompt);
        }
        let last_token_id = *token_ids.last().unwrap();

        // 2. Embed
        let (input_matrix, _) = crate::compiler::hf_loader::load_embedding_row(
            &self.model_dir, self.hidden_size, last_token_id,
        ).map_err(|e| LocalProviderError::EmbedFailed(format!("{e}")))?;

        // 3. Fast forward pass — just matmul chain, no proving
        let output = crate::aggregation::execute_forward_pass_fast(
            &self.graph, &input_matrix, &self.weights,
        ).map_err(|e| LocalProviderError::ProveFailed(format!("{e}")))?;

        // 4. Project to logits for predicted text
        let predicted_text = crate::compiler::hf_loader::project_to_logits(
            &self.model_dir, &output,
        )
        .ok()
        .and_then(|(tid, _)| self.tokenizer.decode(&[tid], true).ok())
        .unwrap_or_default();

        Ok((predicted_text, token_ids, input_matrix))
    }

    /// Batched proving — prove multiple tokens in one pass.
    ///
    /// Embeds all token IDs into a `(N, hidden_size)` matrix and runs ONE
    /// forward pass + GKR proof for all N tokens together. This is the
    /// high-throughput path: at N=10000 on H100, throughput reaches 80+ tok/s.
    pub fn prove_batch(
        &self,
        token_ids: &[u32],
    ) -> Result<(M31Matrix, crate::aggregation::AggregatedModelProofOnChain, u64), LocalProviderError> {
        let t_start = std::time::Instant::now();

        // Embed all tokens into (N, hidden_size) matrix
        let input_matrix = crate::compiler::hf_loader::load_embedding_batch(
            &self.model_dir, self.hidden_size, token_ids,
        ).map_err(|e| LocalProviderError::EmbedFailed(format!("{e}")))?;

        eprintln!(
            "[local] Batched prove: {} tokens, input shape ({}, {})",
            token_ids.len(), input_matrix.rows, input_matrix.cols
        );

        // Reshape graph for batch size
        let batch_graph = self.graph.with_seq_len(token_ids.len());

        // Run full proving pipeline with weight cache (skips weight commitment recomputation)
        let proof = crate::aggregation::prove_model_pure_gkr_auto_with_cache(
            &batch_graph, &input_matrix, &self.weights,
            self.weight_cache.as_ref(), None,
        ).map_err(|e| LocalProviderError::ProveFailed(format!("{e}")))?;

        let prove_time_ms = t_start.elapsed().as_millis() as u64;
        let tok_per_sec = token_ids.len() as f64 / (prove_time_ms as f64 / 1000.0);
        eprintln!(
            "[local] Batch proved: {} tokens in {:.1}s ({:.1} tok/s)",
            token_ids.len(), prove_time_ms as f64 / 1000.0, tok_per_sec
        );

        Ok((proof.execution.output.clone(), proof, prove_time_ms))
    }

    /// Traced inference — captures ForwardPassResult for deferred proving.
    ///
    /// Runs the full forward pass with trace capture, returns both the fast
    /// prediction and the captured `ForwardPassResult` for background proving.
    /// The background prover can then call `prove_from_forward_result()` instead
    /// of re-executing the forward pass.
    pub fn infer_traced(
        &self,
        prompt: &str,
    ) -> Result<(String, Vec<u32>, M31Matrix, Option<crate::aggregation::ForwardPassResult>), LocalProviderError> {
        let encoding = self.tokenizer.encode(prompt, false)
            .map_err(|e| LocalProviderError::TokenizeFailed(format!("{e}")))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();
        if token_ids.is_empty() {
            return Err(LocalProviderError::EmptyPrompt);
        }
        let last_token_id = *token_ids.last().unwrap();

        let (input_matrix, _) = crate::compiler::hf_loader::load_embedding_row(
            &self.model_dir, self.hidden_size, last_token_id,
        ).map_err(|e| LocalProviderError::EmbedFailed(format!("{e}")))?;

        // Run traced execution — captures ForwardPassResult via thread-local
        let (_output, fwd) = crate::aggregation::execute_forward_pass_traced(
            &self.graph, &input_matrix, &self.weights,
        ).map_err(|e| LocalProviderError::ProveFailed(format!("{e}")))?;

        let predicted_text = crate::compiler::hf_loader::project_to_logits(
            &self.model_dir, &fwd.output,
        )
        .ok()
        .and_then(|(tid, _)| self.tokenizer.decode(&[tid], true).ok())
        .unwrap_or_default();

        Ok((predicted_text, token_ids, input_matrix, Some(fwd)))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LocalProviderError {
    #[error("Failed to load model: {0}")]
    LoadFailed(String),
    #[error("Tokenization failed: {0}")]
    TokenizeFailed(String),
    #[error("Prompt produced zero tokens")]
    EmptyPrompt,
    #[error("Embedding lookup failed: {0}")]
    EmbedFailed(String),
    #[error("Proving failed: {0}")]
    ProveFailed(String),
}
