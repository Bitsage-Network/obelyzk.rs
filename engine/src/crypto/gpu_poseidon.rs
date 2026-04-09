//! GPU Poseidon Channel — Fiat-Shamir on-device for zero CPU round-trips.
//!
//! Manages the Poseidon Hades permutation state on GPU, enabling the
//! sumcheck loop to run entirely on-device without CPU synchronization.
//!
//! Usage:
//! ```ignore
//! let mut gpu_ch = GpuPoseidonChannel::from_cpu(&cpu_channel, &executor)?;
//! // ... sumcheck loop with gpu_ch.mix_and_draw_gpu() ...
//! let cpu_channel = gpu_ch.into_cpu()?;
//! ```

#[cfg(feature = "cuda-runtime")]
use std::sync::Arc;

#[cfg(feature = "cuda-runtime")]
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, LaunchAsync, LaunchConfig};

#[cfg(feature = "cuda-runtime")]
use crate::crypto::poseidon_channel::PoseidonChannel;
#[cfg(feature = "cuda-runtime")]
use crate::crypto::poseidon_constants;

/// GPU-resident Poseidon Fiat-Shamir channel.
///
/// Holds the channel state (digest + n_draws) and round constants on GPU.
/// All mix/draw operations run on-device via the Hades CUDA kernel.
#[cfg(feature = "cuda-runtime")]
pub struct GpuPoseidonChannel {
    device: Arc<CudaDevice>,
    /// Channel state on GPU: [digest_w0..w7, n_draws, hash_count_lo, hash_count_hi] = 11 u32
    d_state: CudaSlice<u32>,
    /// Round constants on GPU: 273 × 8 u32 = 2184 u32
    d_round_constants: CudaSlice<u32>,
    /// Compiled Poseidon permutation kernel
    poseidon_fn: CudaFunction,
    /// Number of hashes performed (tracked for CPU sync)
    hash_count: u64,
}

#[cfg(feature = "cuda-runtime")]
impl GpuPoseidonChannel {
    /// Create from a CPU channel — uploads state + round constants to GPU.
    pub fn from_cpu(
        cpu_channel: &PoseidonChannel,
        device: &Arc<CudaDevice>,
    ) -> Result<Self, GpuPoseidonError> {
        // Compile Poseidon kernel
        let kernel_src = include_str!("../cuda/poseidon_hades.cu");
        let ptx = cudarc::nvrtc::compile_ptx(kernel_src)
            .map_err(|e| GpuPoseidonError::KernelCompile(format!("{e:?}")))?;

        device.load_ptx(ptx, "poseidon", &["poseidon_permute_kernel"])
            .map_err(|e| GpuPoseidonError::KernelCompile(format!("{e:?}")))?;

        let poseidon_fn = device.get_func("poseidon", "poseidon_permute_kernel")
            .ok_or_else(|| GpuPoseidonError::KernelCompile("poseidon_permute_kernel not found".into()))?;

        // Upload round constants
        let (rc_u32, _n_rounds, _n_per_round) = poseidon_constants::get_round_constants_u32();
        let d_round_constants = device.htod_sync_copy(&rc_u32)
            .map_err(|e| GpuPoseidonError::Memory(format!("upload round constants: {e:?}")))?;

        // Upload channel state
        let state_u32 = cpu_channel.to_gpu_state();
        let d_state = device.htod_sync_copy(&state_u32)
            .map_err(|e| GpuPoseidonError::Memory(format!("upload state: {e:?}")))?;

        Ok(Self {
            device: Arc::clone(device),
            d_state,
            d_round_constants,
            poseidon_fn,
            hash_count: cpu_channel.hash_count,
        })
    }

    /// Run Poseidon Hades permutation on GPU.
    ///
    /// Operates on `d_state[0..24]` (3 felt252 elements = 24 u32 words).
    /// Modifies state in-place.
    pub fn permute_gpu(&mut self) -> Result<(), GpuPoseidonError> {
        unsafe {
            self.poseidon_fn.clone().launch(
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1), // Single thread — Poseidon is sequential
                    shared_mem_bytes: 0,
                },
                (
                    &mut self.d_state,
                    &self.d_round_constants,
                    4u32,  // n_full_first (first half of 8 full rounds)
                    83u32, // n_partial
                    4u32,  // n_full_last (second half of 8 full rounds)
                ),
            ).map_err(|e| GpuPoseidonError::Kernel(format!("permute: {e:?}")))?;
        }
        self.hash_count += 1;
        Ok(())
    }

    /// Download GPU state back to CPU channel.
    pub fn into_cpu(self) -> Result<PoseidonChannel, GpuPoseidonError> {
        let mut state_u32 = vec![0u32; 11];
        self.device.dtoh_sync_copy_into(&self.d_state, &mut state_u32)
            .map_err(|e| GpuPoseidonError::Memory(format!("download state: {e:?}")))?;

        Ok(PoseidonChannel::from_gpu_state(&state_u32, self.hash_count))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GpuPoseidonError {
    #[error("GPU Poseidon kernel compilation: {0}")]
    KernelCompile(String),
    #[error("GPU Poseidon kernel execution: {0}")]
    Kernel(String),
    #[error("GPU memory: {0}")]
    Memory(String),
}
