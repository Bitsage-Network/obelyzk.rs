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

use crate::error::{CairoProveError, Result};

/// Extracts artifacts from a finished cairo runner, to later be used for proving.
///
/// Uses `bootloader_context()` by default (all 11 builtins present), which is required for
/// STARK-in-STARK recursion where the Cairo verifier expects all 11 segments. For standalone
/// Cairo programs that declare fewer builtins, overrides with actual builtins to match the
/// AP stack layout — these programs cannot be used for STARK-in-STARK.
///
/// # Errors
///
/// Returns an error if the adapter fails to process the runner.
pub fn prover_input_from_runner(runner: &CairoRunner) -> Result<ProverInput> {
    info!("Generating input for the prover...");
    let mut input = adapt(runner)
        .map_err(|e| CairoProveError::AdapterError(format!("{:?}", e)))?;

    // If the program declares fewer than 11 builtins, override bootloader_context
    // with actual builtins. This is needed for standalone programs (JSON/verify path)
    // but means the proof cannot be used for STARK-in-STARK recursion.
    let builtins: Vec<BuiltinName> = runner.get_program_builtins().to_vec();
    if builtins.len() < 11 {
        input.public_segment_context = PublicSegmentContext::new(&builtins);
    }

    info!("Input for the prover generated successfully.");
    Ok(input)
}

/// Deduces the preprocessed trace variant needed for the specific execution, and proves.
///
/// # Errors
///
/// Returns an error if proof generation fails.
pub fn prove(input: ProverInput, pcs_config: PcsConfig) -> Result<CairoProof<Blake2sMerkleHasher>> {
    // Currently there are two variants of the preprocessed trace:
    // - Canonical: Pedersen is included in the program.
    // - CanonicalWithoutPedersen: Pedersen is not included in the program.
    // We deduce the variant based on weather the pedersen builtin is included in the program.
    let preprocessed_trace = match input.public_segment_context[1] {
        true => PreProcessedTraceVariant::Canonical,
        false => PreProcessedTraceVariant::CanonicalWithoutPedersen,
    };
    prove_inner(input, preprocessed_trace, pcs_config)
}

/// Prove with GPU backend when cuda-runtime feature is enabled.
///
/// NOTE: GPU proving requires our custom stwo-gpu backend. The upstream v1.2.2
/// prover does not include prove_cairo_gpu. For now, this falls back to CPU.
/// GPU support will be re-integrated once stwo-gpu is synced with v1.2.2.
#[cfg(feature = "cuda-runtime")]
pub fn prove_gpu(input: ProverInput, pcs_config: PcsConfig) -> Result<CairoProof<Blake2sMerkleHasher>> {
    log::warn!("GPU proving not yet available with stwo_cairo_prover v1.2.2 — falling back to CPU");
    prove(input, pcs_config)
}

fn prove_inner(
    input: ProverInput,
    preprocessed_trace: PreProcessedTraceVariant,
    pcs_config: PcsConfig,
) -> Result<CairoProof<Blake2sMerkleHasher>> {
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
