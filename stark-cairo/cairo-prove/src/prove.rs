use cairo_air::CairoProof;
use cairo_vm::types::builtin_name::BuiltinName;
use cairo_vm::vm::runners::cairo_runner::CairoRunner;
use log::info;
use stwo_cairo_adapter::adapter::adapt;
use stwo_cairo_adapter::{ProverInput, PublicSegmentContext};
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
use stwo_cairo_prover::prover::{prove_cairo, ChannelHash, ProverParameters};
use stwo_cairo_prover::stwo::core::pcs::PcsConfig;
use stwo_cairo_prover::stwo::core::vcs_lifted::blake2_merkle::{
    Blake2sMerkleChannel, Blake2sMerkleHasher,
};
use stwo_cairo_prover::stwo::core::vcs_lifted::poseidon252_merkle::{
    Poseidon252MerkleChannel, Poseidon252MerkleHasher,
};

use crate::error::{CairoProveError, Result};

/// Extracts artifacts from a finished cairo runner, to later be used for proving.
pub fn prover_input_from_runner(runner: &CairoRunner) -> Result<ProverInput> {
    info!("Generating input for the prover...");
    let mut input = adapt(runner)
        .map_err(|e| CairoProveError::AdapterError(format!("{:?}", e)))?;

    let builtins: Vec<BuiltinName> = runner.get_program_builtins().to_vec();
    if builtins.len() < 11 {
        input.public_segment_context = PublicSegmentContext::new(&builtins);
    }

    info!("Input for the prover generated successfully.");
    Ok(input)
}

/// Prove with Blake2s Merkle channel (fast prover, large verifier in Cairo).
pub fn prove(input: ProverInput, pcs_config: PcsConfig) -> Result<CairoProof<Blake2sMerkleHasher>> {
    let preprocessed_trace = PreProcessedTraceVariant::Canonical;
    let prover_params = ProverParameters {
        channel_hash: ChannelHash::Blake2s,
        channel_salt: 0,
        pcs_config,
        preprocessed_trace,
        store_polynomials_coefficients: false,
        include_all_preprocessed_columns: false,
    };

    prove_cairo::<Blake2sMerkleChannel>(input, prover_params)
        .map_err(|e| CairoProveError::ProofGeneration(format!("{:?}", e)))
}

/// Prove with Poseidon252 Merkle channel (slower prover, but the Cairo verifier
/// is ~60% smaller because Poseidon is a native builtin). This is the path for
/// on-chain recursive verification — the Level 2 proof is small enough for
/// Starknet calldata (~19K felts = 4 transactions).
pub fn prove_poseidon(
    input: ProverInput,
    pcs_config: PcsConfig,
) -> Result<CairoProof<Poseidon252MerkleHasher>> {
    info!("[Poseidon252] Using native Poseidon channel for on-chain recursive path.");
    // Poseidon252 verifier uses CanonicalWithoutPedersen preprocessed trace
    let preprocessed_trace = PreProcessedTraceVariant::CanonicalWithoutPedersen;
    let prover_params = ProverParameters {
        channel_hash: ChannelHash::Poseidon252,
        channel_salt: 0,
        pcs_config,
        preprocessed_trace,
        store_polynomials_coefficients: false,
        include_all_preprocessed_columns: false,
    };

    prove_cairo::<Poseidon252MerkleChannel>(input, prover_params)
        .map_err(|e| CairoProveError::ProofGeneration(format!("{:?}", e)))
}

#[cfg(feature = "gpu")]
pub fn prove_gpu(input: ProverInput, pcs_config: PcsConfig) -> Result<CairoProof<Blake2sMerkleHasher>> {
    info!("[GPU] cuda-runtime feature enabled.");
    prove(input, pcs_config)
}

/// Prove with 160-bit security for on-chain recursive verification.
/// Uses Poseidon252 channel + blowup=32 + 28 queries.
/// The recursive proof fits in ~2800 felts (under Starknet's 5000-felt TX limit).
pub fn prove_recursive_160(
    input: ProverInput,
) -> Result<CairoProof<Poseidon252MerkleHasher>> {
    info!("[160-bit recursive] Poseidon252 channel, blowup=32, 28 queries.");
    let pcs_config = PcsConfig {
        pow_bits: 20,
        fri_config: stwo_cairo_prover::stwo::core::fri::FriConfig::new(0, 5, 28, 1),
        lifting_log_size: None,
    };
    prove_poseidon(input, pcs_config)
}

#[cfg(feature = "gpu")]
pub fn prove_gpu_poseidon(
    input: ProverInput,
    pcs_config: PcsConfig,
) -> Result<CairoProof<Poseidon252MerkleHasher>> {
    info!("[GPU+Poseidon252] CUDA + native Poseidon channel for on-chain recursive path.");
    prove_poseidon(input, pcs_config)
}

#[cfg(feature = "gpu")]
pub fn prove_gpu_recursive_160(
    input: ProverInput,
) -> Result<CairoProof<Poseidon252MerkleHasher>> {
    info!("[GPU+160-bit] CUDA + Poseidon252, blowup=32, 28 queries.");
    prove_recursive_160(input)
}
