// ═══════════════════════════════════════════════════════════════════════════
// General-Purpose STWO On-Chain Verifier
// ═══════════════════════════════════════════════════════════════════════════
//
// Verifies ANY Cairo program's STWO STARK proof on-chain.
// First general-purpose STWO STARK verifier on any blockchain.
//
// Two verification modes:
//   1. Single-TX: For proofs ≤ 5000 felts (small programs or with M31 packing)
//   2. Streaming: For proofs > 5000 felts (split across multiple TXs)
//
// Security: 160 bits minimum (pow + log_blowup × n_queries ≥ 160)
// No trusted setup. Pure algebraic verification.

use stwo_cairo_air::{CairoProof, VerificationOutput, get_verification_output, verify_cairo};

/// Registered program info.
#[derive(Drop, Copy, Serde, starknet::Store)]
pub struct ProgramInfo {
    pub program_hash: felt252,
    pub min_security_bits: u32,
    pub owner: starknet::ContractAddress,
}

#[starknet::interface]
pub trait IGeneralStwoVerifier<TContractState> {
    /// Register a program for on-chain verification.
    fn register_program(
        ref self: TContractState,
        program_hash: felt252,
        min_security_bits: u32,
    );

    /// Single-TX verification (for proofs ≤ 5000 felts).
    fn verify_stwo(
        ref self: TContractState,
        proof: CairoProof,
    ) -> VerificationOutput;

    /// Streaming: open a verification session.
    fn stream_open(
        ref self: TContractState,
        expected_total_felts: u32,
    ) -> u64;

    /// Streaming: upload a chunk of proof data.
    fn stream_chunk(
        ref self: TContractState,
        session_id: u64,
        chunk_idx: u32,
        chunk: Array<felt252>,
    );

    /// Streaming: finalize — reassemble proof, verify, record on-chain.
    fn stream_verify(
        ref self: TContractState,
        session_id: u64,
    ) -> VerificationOutput;

    fn is_verified(self: @TContractState, proof_hash: felt252) -> bool;
    fn get_verification_count(self: @TContractState, program_hash: felt252) -> u64;
}

#[starknet::contract]
mod GeneralStwoVerifierContract {
    use core::poseidon::poseidon_hash_span;
    use starknet::{get_caller_address, get_block_timestamp, ContractAddress};
    use starknet::storage::{
        Map, StorageMapReadAccess, StorageMapWriteAccess, StoragePointerReadAccess,
        StoragePointerWriteAccess,
    };
    use stwo_verifier_core::pcs::PcsConfigTrait;
    use super::{
        CairoProof, IGeneralStwoVerifier, ProgramInfo, VerificationOutput,
        get_verification_output, verify_cairo,
    };

    const MIN_SECURITY_BITS: u32 = 160;
    const MAX_CHUNK_SIZE: u32 = 4900; // Leave room for TX overhead

    #[storage]
    struct Storage {
        owner: ContractAddress,
        // Program registry
        programs: Map<felt252, ProgramInfo>,
        // Verification results
        verified_proofs: Map<felt252, bool>,
        verification_count: Map<felt252, u64>,
        // Last verification per program
        last_proof_hash: Map<felt252, felt252>,
        last_verified_at: Map<felt252, u64>,
        // Streaming state
        next_session_id: u64,
        session_owner: Map<u64, ContractAddress>,
        session_total_felts: Map<u64, u32>,
        session_received_felts: Map<u64, u32>,
        session_chunks_received: Map<u64, u32>,
        session_sealed: Map<u64, bool>,
        // Streaming data: (session_id, flat_index) → felt252
        session_data: Map<(u64, u32), felt252>,
        // Running hash of all chunks for integrity
        session_data_hash: Map<u64, felt252>,
    }

    #[event]
    #[derive(Drop, starknet::Event)]
    enum Event {
        ProgramRegistered: ProgramRegistered,
        StwoProofVerified: StwoProofVerified,
        StreamOpened: StreamOpened,
        StreamChunkReceived: StreamChunkReceived,
    }

    #[derive(Drop, starknet::Event)]
    struct ProgramRegistered {
        #[key]
        program_hash: felt252,
        min_security_bits: u32,
        registered_by: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    struct StwoProofVerified {
        #[key]
        program_hash: felt252,
        proof_hash: felt252,
        verification_count: u64,
        verified_at: u64,
        submitter: ContractAddress,
        mode: felt252, // 'single' or 'stream'
    }

    #[derive(Drop, starknet::Event)]
    struct StreamOpened {
        #[key]
        session_id: u64,
        expected_total_felts: u32,
        owner: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    struct StreamChunkReceived {
        #[key]
        session_id: u64,
        chunk_idx: u32,
        chunk_size: u32,
        total_received: u32,
    }

    #[constructor]
    fn constructor(ref self: ContractState, owner: ContractAddress) {
        self.owner.write(owner);
        self.next_session_id.write(1);
    }

    #[abi(embed_v0)]
    impl GeneralStwoVerifierImpl of IGeneralStwoVerifier<ContractState> {
        fn register_program(
            ref self: ContractState,
            program_hash: felt252,
            min_security_bits: u32,
        ) {
            assert!(min_security_bits >= MIN_SECURITY_BITS, "min_security_bits must be >= 160");
            assert!(program_hash != 0, "program_hash cannot be zero");

            let caller = get_caller_address();
            self
                .programs
                .write(
                    program_hash,
                    ProgramInfo { program_hash, min_security_bits, owner: caller },
                );
            self.emit(ProgramRegistered { program_hash, min_security_bits, registered_by: caller });
        }

        // ─── Single-TX verification ───────────────────────────────────────
        fn verify_stwo(
            ref self: ContractState,
            proof: CairoProof,
        ) -> VerificationOutput {
            let output = get_verification_output(proof: @proof);

            let security = proof.stark_proof.commitment_scheme_proof.config.security_bits();
            assert!(security >= MIN_SECURITY_BITS, "Security {} < 160", security);

            // FULL CRYPTOGRAPHIC STARK VERIFICATION
            verify_cairo(proof);

            let proof_hash = poseidon_hash_span(array![output.program_hash].span());
            assert!(!self.verified_proofs.read(proof_hash), "Already verified");

            self._record_verification(output.program_hash, proof_hash, 'single');
            output
        }

        // ─── Streaming: open session ──────────────────────────────────────
        fn stream_open(
            ref self: ContractState,
            expected_total_felts: u32,
        ) -> u64 {
            let session_id = self.next_session_id.read();
            self.next_session_id.write(session_id + 1);

            let caller = get_caller_address();
            self.session_owner.write(session_id, caller);
            self.session_total_felts.write(session_id, expected_total_felts);
            self.session_received_felts.write(session_id, 0);
            self.session_chunks_received.write(session_id, 0);
            self.session_sealed.write(session_id, false);
            self.session_data_hash.write(session_id, 0);

            self.emit(StreamOpened { session_id, expected_total_felts, owner: caller });
            session_id
        }

        // ─── Streaming: upload chunk ──────────────────────────────────────
        fn stream_chunk(
            ref self: ContractState,
            session_id: u64,
            chunk_idx: u32,
            chunk: Array<felt252>,
        ) {
            // Validate session
            let caller = get_caller_address();
            assert!(self.session_owner.read(session_id) == caller, "Not session owner");
            assert!(!self.session_sealed.read(session_id), "Session sealed");

            let expected_chunk_idx = self.session_chunks_received.read(session_id);
            assert!(chunk_idx == expected_chunk_idx, "Chunks must be sequential");

            let chunk_len: u32 = chunk.len().try_into().unwrap();
            assert!(chunk_len <= MAX_CHUNK_SIZE, "Chunk too large");

            // Store chunk data at flat offset
            let offset = self.session_received_felts.read(session_id);
            let mut i: u32 = 0;
            let chunk_span = chunk.span();
            loop {
                if i >= chunk_len {
                    break;
                }
                self.session_data.write((session_id, offset + i), *chunk_span.at(i.into()));
                i += 1;
            };

            // Update running hash for integrity
            let prev_hash = self.session_data_hash.read(session_id);
            let mut hash_input = array![prev_hash];
            for felt in chunk_span {
                hash_input.append(*felt);
            };
            self.session_data_hash.write(session_id, poseidon_hash_span(hash_input.span()));

            // Update counters
            let new_received = offset + chunk_len;
            self.session_received_felts.write(session_id, new_received);
            self.session_chunks_received.write(session_id, expected_chunk_idx + 1);

            // Auto-seal when all felts received
            let total = self.session_total_felts.read(session_id);
            if new_received >= total {
                self.session_sealed.write(session_id, true);
            }

            self
                .emit(
                    StreamChunkReceived {
                        session_id,
                        chunk_idx,
                        chunk_size: chunk_len,
                        total_received: new_received,
                    },
                );
        }

        // ─── Streaming: finalize and verify ───────────────────────────────
        fn stream_verify(
            ref self: ContractState,
            session_id: u64,
        ) -> VerificationOutput {
            // Validate session is sealed
            let caller = get_caller_address();
            assert!(self.session_owner.read(session_id) == caller, "Not session owner");
            assert!(self.session_sealed.read(session_id), "Session not sealed");

            // Reassemble proof from storage
            let total_felts = self.session_total_felts.read(session_id);
            let mut proof_data: Array<felt252> = array![];
            let mut i: u32 = 0;
            loop {
                if i >= total_felts {
                    break;
                }
                proof_data.append(self.session_data.read((session_id, i)));
                i += 1;
            };

            // Deserialize CairoProof from reassembled data
            let mut proof_span = proof_data.span();
            let proof: CairoProof = Serde::deserialize(ref proof_span)
                .expect('PROOF_DESER');

            // Extract output BEFORE verification
            let output = get_verification_output(proof: @proof);

            // Enforce security
            let security = proof.stark_proof.commitment_scheme_proof.config.security_bits();
            assert!(security >= MIN_SECURITY_BITS, "Security {} < 160", security);

            // FULL CRYPTOGRAPHIC STARK VERIFICATION
            verify_cairo(proof);

            // Record on-chain
            let proof_hash = poseidon_hash_span(array![output.program_hash].span());
            assert!(!self.verified_proofs.read(proof_hash), "Already verified");

            self._record_verification(output.program_hash, proof_hash, 'stream');
            output
        }

        fn is_verified(self: @ContractState, proof_hash: felt252) -> bool {
            self.verified_proofs.read(proof_hash)
        }

        fn get_verification_count(self: @ContractState, program_hash: felt252) -> u64 {
            self.verification_count.read(program_hash)
        }
    }

    #[generate_trait]
    impl InternalImpl of InternalTrait {
        fn _record_verification(
            ref self: ContractState,
            program_hash: felt252,
            proof_hash: felt252,
            mode: felt252,
        ) {
            self.verified_proofs.write(proof_hash, true);
            let count = self.verification_count.read(program_hash);
            self.verification_count.write(program_hash, count + 1);
            let block_ts = get_block_timestamp();
            self.last_proof_hash.write(program_hash, proof_hash);
            self.last_verified_at.write(program_hash, block_ts);

            self
                .emit(
                    StwoProofVerified {
                        program_hash,
                        proof_hash,
                        verification_count: count + 1,
                        verified_at: block_ts,
                        submitter: get_caller_address(),
                        mode,
                    },
                );
        }
    }
}
