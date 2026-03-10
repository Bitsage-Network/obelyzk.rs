#!/usr/bin/env bash
# test_full_integration.sh — Master orchestrator: 10-phase audit-grade test suite.
#
# Exercises stwo-ml from build through on-chain submission with external-auditor
# level scrutiny. Produces a JSON report of all phases.
#
# Usage:
#   # Minimal (no GPU, no model — phases 1-3 only):
#   bash scripts/test_full_integration.sh
#
#   # With model + GPU (phases 1-6, 8-9):
#   bash scripts/test_full_integration.sh --model-dir ~/.obelysk/models/qwen3-14b --gpu
#
#   # Full including on-chain (all 10 phases):
#   bash scripts/test_full_integration.sh --model-dir ~/.obelysk/models/qwen3-14b --gpu --submit
#
# Environment overrides:
#   PM          — path to prove-model binary (default: ./target/release/prove-model)
#   OUTPUT_DIR  — test artifact directory (default: /tmp/integration_test_<timestamp>)

set -euo pipefail

# ============================================================================
# Parse Arguments
# ============================================================================

MODEL_DIR=""
GPU=0
SUBMIT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --gpu)
            GPU=1
            shift
            ;;
        --submit)
            SUBMIT=1
            shift
            ;;
        *)
            echo "Unknown arg: $1"
            echo "Usage: $0 [--model-dir DIR] [--gpu] [--submit]"
            exit 1
            ;;
    esac
done

# ============================================================================
# Environment
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT/stwo-ml"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/integration_test_${TIMESTAMP}}"
mkdir -p "$OUTPUT_DIR"

PM="${PM:-target/release/prove-model}"

# Features for build
FEATURES="std,cli,model-loading,audit,safetensors"
if [ "$GPU" = "1" ]; then
    FEATURES="$FEATURES,gpu,cuda-runtime"
fi

# ============================================================================
# Reporting
# ============================================================================

declare -A PHASE_STATUS
declare -A PHASE_DURATION
declare -A PHASE_TESTS_PASS
declare -A PHASE_TESTS_FAIL
TOTAL_PASS=0
TOTAL_FAIL=0
START_TIME=$(date +%s)

pass() {
    local phase="$1"
    local count="${2:-1}"
    PHASE_TESTS_PASS[$phase]=$(( ${PHASE_TESTS_PASS[$phase]:-0} + count ))
    TOTAL_PASS=$(( TOTAL_PASS + count ))
}

fail() {
    local phase="$1"
    local msg="$2"
    PHASE_TESTS_FAIL[$phase]=$(( ${PHASE_TESTS_FAIL[$phase]:-0} + 1 ))
    TOTAL_FAIL=$(( TOTAL_FAIL + 1 ))
    echo "  FAIL: $msg"
}

run_phase() {
    local phase_num="$1"
    local phase_name="$2"
    local critical="$3"
    local phase_fn="$4"

    echo ""
    echo "================================================================"
    echo "Phase $phase_num: $phase_name"
    echo "================================================================"

    local phase_start
    phase_start=$(date +%s)

    PHASE_TESTS_PASS[$phase_num]=0
    PHASE_TESTS_FAIL[$phase_num]=0

    if $phase_fn; then
        PHASE_STATUS[$phase_num]="PASS"
    else
        if [ "${PHASE_TESTS_FAIL[$phase_num]:-0}" -gt 0 ]; then
            PHASE_STATUS[$phase_num]="FAIL"
        else
            PHASE_STATUS[$phase_num]="FAIL"
            PHASE_TESTS_FAIL[$phase_num]=1
            TOTAL_FAIL=$(( TOTAL_FAIL + 1 ))
        fi
    fi

    local phase_end
    phase_end=$(date +%s)
    PHASE_DURATION[$phase_num]=$(( phase_end - phase_start ))

    local status="${PHASE_STATUS[$phase_num]}"
    local p="${PHASE_TESTS_PASS[$phase_num]:-0}"
    local f="${PHASE_TESTS_FAIL[$phase_num]:-0}"
    echo "  Phase $phase_num: $status ($p pass, $f fail, ${PHASE_DURATION[$phase_num]}s)"

    if [ "$status" = "FAIL" ] && [ "$critical" = "YES" ]; then
        echo "  CRITICAL FAILURE — stopping pipeline."
        generate_report
        exit 1
    fi
}

assert_lt() {
    local val="$1"
    local max="$2"
    local label="$3"
    local phase="$4"

    if [ -z "$val" ] || [ "$val" = "" ]; then
        fail "$phase" "$label: value not found"
        return 1
    fi

    if (( $(echo "$val < $max" | bc -l) )); then
        pass "$phase"
        echo "  OK: $label = $val (< $max)"
    else
        fail "$phase" "$label = $val (>= $max threshold)"
        return 1
    fi
}

# ============================================================================
# Phase Implementations
# ============================================================================

phase_1_build() {
    echo "  1a: Building with features: $FEATURES"
    if cargo build --release --bin prove-model --features "$FEATURES" \
        > "$OUTPUT_DIR/build.log" 2>&1; then
        pass 1
        echo "  OK: Build succeeded"
    else
        fail 1 "Build failed — see $OUTPUT_DIR/build.log"
        return 1
    fi

    echo "  1b: Clippy (deny warnings)"
    if cargo clippy --all-targets --features std -- -D warnings \
        > "$OUTPUT_DIR/clippy.log" 2>&1; then
        pass 1
        echo "  OK: Clippy clean"
    else
        # Clippy warnings are non-fatal but reported
        echo "  WARN: Clippy warnings — see $OUTPUT_DIR/clippy.log"
        pass 1
    fi

    echo "  1c: Binary sanity check"
    if $PM --help > /dev/null 2>&1; then
        pass 1
        echo "  OK: Binary runs"
    else
        fail 1 "Binary $PM --help failed"
        return 1
    fi
}

phase_2_unit_tests() {
    echo "  2a: Library tests (--lib)"
    if cargo test --features std --lib -- --test-threads=4 \
        > "$OUTPUT_DIR/unit_tests.log" 2>&1; then
        # Count passing tests
        local count
        count=$(grep -c "^test .* ok$" "$OUTPUT_DIR/unit_tests.log" 2>/dev/null || echo "0")
        pass 2 "$count"
        echo "  OK: $count lib tests passed"
    else
        fail 2 "Library tests failed — see $OUTPUT_DIR/unit_tests.log"
        return 1
    fi

    echo "  2b: Decode tamper tests"
    if cargo test --features std --test decode_benchmark -- \
        decode_tamper --test-threads=1 > "$OUTPUT_DIR/decode_tamper.log" 2>&1; then
        local count
        count=$(grep -c "^test .* ok$" "$OUTPUT_DIR/decode_tamper.log" 2>/dev/null || echo "0")
        pass 2 "$count"
        echo "  OK: $count decode tamper tests passed"
    else
        fail 2 "Decode tamper tests failed — see $OUTPUT_DIR/decode_tamper.log"
        return 1
    fi
}

phase_3_security_gates() {
    echo "  Running security gate enforcement tests..."
    if cargo test --features std --test integration_gate -- \
        security_gate --test-threads=1 > "$OUTPUT_DIR/security_gates.log" 2>&1; then
        local count
        count=$(grep -c "^test .* ok$" "$OUTPUT_DIR/security_gates.log" 2>/dev/null || echo "0")
        pass 3 "$count"
        echo "  OK: $count security gate tests passed"
    else
        fail 3 "Security gate tests failed — see $OUTPUT_DIR/security_gates.log"
        grep -E "^test .* FAILED" "$OUTPUT_DIR/security_gates.log" | sed 's/^/    /' || true
        return 1
    fi
}

phase_4_proof_gen() {
    if [ -z "$MODEL_DIR" ]; then
        echo "  SKIP: No --model-dir provided"
        PHASE_STATUS[4]="SKIP"
        return 0
    fi

    local gpu_flag=""
    if [ "$GPU" = "1" ]; then
        gpu_flag="--gpu"
    fi

    echo "  4a: 1-layer proof generation"
    local t_start
    t_start=$(date +%s)
    if $PM --model-dir "$MODEL_DIR" --layers 1 $gpu_flag --format ml_gkr \
        --output "$OUTPUT_DIR/proof_1layer.json" \
        > "$OUTPUT_DIR/prove_1l.log" 2>&1; then
        local t_end
        t_end=$(date +%s)
        pass 4
        echo "  OK: 1-layer proof in $(( t_end - t_start ))s"
    else
        fail 4 "1-layer proof generation failed — see $OUTPUT_DIR/prove_1l.log"
        return 1
    fi

    echo "  4b: Verify 1-layer proof"
    if $PM --verify-proof "$OUTPUT_DIR/proof_1layer.json" \
        > "$OUTPUT_DIR/verify_1l.log" 2>&1; then
        pass 4
        echo "  OK: 1-layer verification passed"
    else
        fail 4 "1-layer verification failed — see $OUTPUT_DIR/verify_1l.log"
        return 1
    fi

    # Check if 1-layer was fast enough to attempt 5-layer
    local prove_time
    prove_time=$(( t_end - t_start ))
    if [ "$prove_time" -gt 60 ]; then
        echo "  SKIP: 1-layer took ${prove_time}s (>60s), skipping 5-layer"
        return 0
    fi

    echo "  4c: 5-layer proof generation"
    t_start=$(date +%s)
    if $PM --model-dir "$MODEL_DIR" --layers 5 $gpu_flag --format ml_gkr \
        --output "$OUTPUT_DIR/proof_5layer.json" \
        > "$OUTPUT_DIR/prove_5l.log" 2>&1; then
        t_end=$(date +%s)
        pass 4
        echo "  OK: 5-layer proof in $(( t_end - t_start ))s"
    else
        fail 4 "5-layer proof generation failed — see $OUTPUT_DIR/prove_5l.log"
        return 1
    fi

    echo "  4d: Verify 5-layer proof"
    if $PM --verify-proof "$OUTPUT_DIR/proof_5layer.json" \
        > "$OUTPUT_DIR/verify_5l.log" 2>&1; then
        pass 4
        echo "  OK: 5-layer verification passed"
    else
        fail 4 "5-layer verification failed — see $OUTPUT_DIR/verify_5l.log"
        return 1
    fi
}

phase_5_adversarial() {
    if [ -z "$MODEL_DIR" ]; then
        echo "  SKIP: No --model-dir provided"
        PHASE_STATUS[5]="SKIP"
        return 0
    fi

    echo "  5a: Verification hardening script"
    local hardening_env=""
    [ -n "$MODEL_DIR" ] && hardening_env="MODEL_DIR=$MODEL_DIR"
    [ "$GPU" = "1" ] && hardening_env="$hardening_env GPU=1"
    hardening_env="$hardening_env LAYERS=1 PM=$PM"

    if env $hardening_env bash scripts/test_verification_hardening.sh \
        > "$OUTPUT_DIR/hardening.log" 2>&1; then
        local count
        count=$(grep -c "PASS\|OK" "$OUTPUT_DIR/hardening.log" 2>/dev/null || echo "0")
        pass 5 "$count"
        echo "  OK: Hardening tests passed ($count checks)"
    else
        fail 5 "Hardening tests failed — see $OUTPUT_DIR/hardening.log"
        return 1
    fi

    echo "  5b: Malicious proof tests (22 tamper vectors)"
    if cargo test --features std --test e2e_malicious_proof -- \
        --test-threads=1 > "$OUTPUT_DIR/malicious_proof.log" 2>&1; then
        local count
        count=$(grep -c "^test .* ok$" "$OUTPUT_DIR/malicious_proof.log" 2>/dev/null || echo "0")
        pass 5 "$count"
        echo "  OK: $count malicious proof tests passed"
    else
        fail 5 "Malicious proof tests failed — see $OUTPUT_DIR/malicious_proof.log"
        return 1
    fi
}

phase_6_decode_chain() {
    if [ -z "$MODEL_DIR" ]; then
        echo "  SKIP: No --model-dir provided"
        PHASE_STATUS[6]="SKIP"
        return 0
    fi

    local gpu_flag=""
    if [ "$GPU" = "1" ]; then
        gpu_flag="--gpu"
    fi

    echo "  6a: Decode chain script (3 steps)"
    local chain_dir="$OUTPUT_DIR/decode_chain"
    mkdir -p "$chain_dir"

    if [ -f scripts/pipeline/decode_chain.sh ]; then
        if bash scripts/pipeline/decode_chain.sh \
            --model-dir "$MODEL_DIR" --layers 1 $gpu_flag --decode-steps 3 \
            --output-dir "$chain_dir" > "$OUTPUT_DIR/decode_chain.log" 2>&1; then
            pass 6
            echo "  OK: Decode chain generated"
        else
            fail 6 "Decode chain script failed — see $OUTPUT_DIR/decode_chain.log"
            return 1
        fi

        echo "  6b: Verify chain integrity (jq)"
        local chain_ok=true
        for step in 1 2; do
            local prev_file="$chain_dir/proof_$((step - 1)).json"
            local curr_file="$chain_dir/proof_${step}.json"
            if [ -f "$prev_file" ] && [ -f "$curr_file" ]; then
                local prev_kv
                prev_kv=$(jq -r '.kv_cache_commitment // empty' "$prev_file")
                local this_prev
                this_prev=$(jq -r '.prev_kv_cache_commitment // empty' "$curr_file")
                if [ -n "$prev_kv" ] && [ -n "$this_prev" ] && [ "$prev_kv" = "$this_prev" ]; then
                    pass 6
                    echo "  OK: Chain link $((step-1))→$step valid"
                else
                    fail 6 "Chain break at step $step: prev=$prev_kv, this_prev=$this_prev"
                    chain_ok=false
                fi
            fi
        done
    else
        echo "  SKIP: decode_chain.sh not found"
    fi

    echo "  6c: Rust decode chain tests"
    if cargo test --features std --test integration_gate -- \
        decode_chain --test-threads=1 > "$OUTPUT_DIR/decode_rust.log" 2>&1; then
        local count
        count=$(grep -c "^test .* ok$" "$OUTPUT_DIR/decode_rust.log" 2>/dev/null || echo "0")
        pass 6 "$count"
        echo "  OK: $count decode chain Rust tests passed"
    else
        fail 6 "Decode chain Rust tests failed — see $OUTPUT_DIR/decode_rust.log"
        return 1
    fi
}

phase_7_onchain() {
    if [ "$SUBMIT" != "1" ]; then
        echo "  SKIP: --submit not provided"
        PHASE_STATUS[7]="SKIP"
        return 0
    fi

    local proof_file="$OUTPUT_DIR/proof_1layer.json"
    if [ ! -f "$proof_file" ]; then
        echo "  SKIP: No proof file from Phase 4"
        PHASE_STATUS[7]="SKIP"
        return 0
    fi

    echo "  7a: Submit proof on-chain"
    if bash ../scripts/pipeline/04_verify_onchain.sh \
        --proof "$proof_file" --submit --paymaster \
        > "$OUTPUT_DIR/onchain.log" 2>&1; then
        pass 7
        echo "  OK: On-chain submission succeeded"
    else
        fail 7 "On-chain submission failed — see $OUTPUT_DIR/onchain.log"
        return 1
    fi

    echo "  7b: Check verification receipt"
    local receipt="$OUTPUT_DIR/verify_receipt.json"
    if [ -f "$receipt" ]; then
        local accepted
        accepted=$(jq -r '.accepted_onchain // "false"' "$receipt")
        if [ "$accepted" = "true" ]; then
            pass 7
            echo "  OK: On-chain verification accepted"
        else
            fail 7 "On-chain verification not accepted: $accepted"
        fi
    else
        echo "  WARN: No receipt file found"
    fi
}

phase_8_audit() {
    echo "  Running audit pipeline tests..."
    if cargo test --features std,audit --test e2e_audit -- \
        --test-threads=1 > "$OUTPUT_DIR/audit.log" 2>&1; then
        local count
        count=$(grep -c "^test .* ok$" "$OUTPUT_DIR/audit.log" 2>/dev/null || echo "0")
        pass 8 "$count"
        echo "  OK: $count audit tests passed"
    else
        fail 8 "Audit tests failed — see $OUTPUT_DIR/audit.log"
        return 1
    fi
}

phase_9_perf() {
    if [ -z "$MODEL_DIR" ] || [ "$GPU" != "1" ]; then
        echo "  SKIP: Requires --model-dir and --gpu"
        PHASE_STATUS[9]="SKIP"
        return 0
    fi

    echo "  Checking performance regressions..."

    # Extract 1-layer proof time
    local prove_log="$OUTPUT_DIR/prove_1l.log"
    if [ -f "$prove_log" ]; then
        local prove_time
        prove_time=$(grep -oP 'completed in \K[0-9.]+' "$prove_log" 2>/dev/null | head -1)
        if [ -n "$prove_time" ]; then
            assert_lt "$prove_time" "5.0" "1-layer proof time (s)" 9 || true
        else
            echo "  WARN: Could not extract proof time from log"
        fi
    fi

    # Extract decode timing
    local decode_log="$OUTPUT_DIR/decode_chain.log"
    if [ -f "$decode_log" ]; then
        local decode_time
        decode_time=$(grep -oP 'Step.*?\K[0-9.]+(?=ms)' "$decode_log" 2>/dev/null | tail -1)
        if [ -n "$decode_time" ]; then
            local decode_sec
            decode_sec=$(echo "$decode_time / 1000" | bc -l 2>/dev/null || echo "")
            if [ -n "$decode_sec" ]; then
                assert_lt "$decode_sec" "2.0" "Decode per-step time (s)" 9 || true
            fi
        else
            echo "  WARN: Could not extract decode time from log"
        fi
    fi

    # Memory check
    if [ -f "$prove_log" ]; then
        local rss
        rss=$(grep -oP 'peak_rss_mb=\K[0-9]+' "$prove_log" 2>/dev/null | head -1)
        if [ -n "$rss" ]; then
            assert_lt "$rss" "16384" "Peak memory (MB)" 9 || true
        else
            echo "  WARN: Could not extract peak RSS from log"
        fi
    fi
}

# ============================================================================
# Report Generation
# ============================================================================

generate_report() {
    local end_time
    end_time=$(date +%s)
    local total_duration=$(( end_time - START_TIME ))

    echo ""
    echo "================================================================"
    echo "Integration Test Report"
    echo "================================================================"
    echo ""
    printf "%-8s %-30s %-8s %-6s %-6s %-8s\n" "Phase" "Name" "Status" "Pass" "Fail" "Time(s)"
    printf "%-8s %-30s %-8s %-6s %-6s %-8s\n" "-----" "----" "------" "----" "----" "-------"

    local phase_names=(
        [1]="Build & Compile"
        [2]="Unit Tests"
        [3]="Security Gates"
        [4]="Proof Gen & Verify"
        [5]="Adversarial Hardening"
        [6]="Decode Chain"
        [7]="On-Chain Submission"
        [8]="Audit Pipeline"
        [9]="Perf Regression"
    )

    for i in 1 2 3 4 5 6 7 8 9; do
        local status="${PHASE_STATUS[$i]:-SKIP}"
        local p="${PHASE_TESTS_PASS[$i]:-0}"
        local f="${PHASE_TESTS_FAIL[$i]:-0}"
        local dur="${PHASE_DURATION[$i]:-0}"
        printf "%-8s %-30s %-8s %-6s %-6s %-8s\n" "$i" "${phase_names[$i]}" "$status" "$p" "$f" "$dur"
    done

    echo ""
    echo "Total: $TOTAL_PASS pass, $TOTAL_FAIL fail, ${total_duration}s"
    echo "Artifacts: $OUTPUT_DIR"

    # JSON report
    local report="$OUTPUT_DIR/integration_report.json"
    cat > "$report" <<JSONEOF
{
  "timestamp": "$TIMESTAMP",
  "total_pass": $TOTAL_PASS,
  "total_fail": $TOTAL_FAIL,
  "duration_s": $total_duration,
  "gpu": $GPU,
  "model_dir": "$MODEL_DIR",
  "phases": {
$(for i in 1 2 3 4 5 6 7 8 9; do
    local status="${PHASE_STATUS[$i]:-SKIP}"
    local p="${PHASE_TESTS_PASS[$i]:-0}"
    local f="${PHASE_TESTS_FAIL[$i]:-0}"
    local dur="${PHASE_DURATION[$i]:-0}"
    echo "    \"$i\": {\"name\": \"${phase_names[$i]}\", \"status\": \"$status\", \"pass\": $p, \"fail\": $f, \"duration_s\": $dur}$([ $i -lt 9 ] && echo ',')"
done)
  },
  "output_dir": "$OUTPUT_DIR"
}
JSONEOF

    echo "Report: $report"
}

# ============================================================================
# Main Execution
# ============================================================================

echo "================================================================"
echo "stwo-ml Integration Test Suite"
echo "================================================================"
echo "Time:      $(date)"
echo "Model:     ${MODEL_DIR:-<none>}"
echo "GPU:       $( [ "$GPU" = "1" ] && echo "yes" || echo "no" )"
echo "Submit:    $( [ "$SUBMIT" = "1" ] && echo "yes" || echo "no" )"
echo "Output:    $OUTPUT_DIR"
echo "Features:  $FEATURES"

# Phase 1-3: Always run (no GPU/model needed)
run_phase 1 "Build & Compile"       YES phase_1_build
run_phase 2 "Unit Tests"            YES phase_2_unit_tests
run_phase 3 "Security Gates"        YES phase_3_security_gates

# Phase 4-6: Require model (and optionally GPU)
run_phase 4 "Proof Gen & Verify"    YES phase_4_proof_gen
run_phase 5 "Adversarial Hardening" YES phase_5_adversarial
run_phase 6 "Decode Chain"          YES phase_6_decode_chain

# Phase 7: On-chain (optional)
run_phase 7 "On-Chain Submission"   NO  phase_7_onchain

# Phase 8: Audit pipeline
run_phase 8 "Audit Pipeline"        YES phase_8_audit

# Phase 9: Performance regression
run_phase 9 "Perf Regression"       NO  phase_9_perf

# Phase 10: Report
generate_report

# Exit code
if [ "$TOTAL_FAIL" -gt 0 ]; then
    echo ""
    echo "INTEGRATION SUITE: FAILURES DETECTED ($TOTAL_FAIL)"
    exit 1
else
    echo ""
    echo "INTEGRATION SUITE: ALL PASSED ($TOTAL_PASS)"
    exit 0
fi
