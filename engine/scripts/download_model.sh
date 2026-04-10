#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# ObelyZK — Download Model
#
# Download any supported model for verifiable inference.
#
# Usage:
#   ./scripts/download_model.sh qwen2.5-14b
#   ./scripts/download_model.sh glm-4-9b
#   ./scripts/download_model.sh minimax-m2.5
#   ./scripts/download_model.sh kimi-k2.5
#   ./scripts/download_model.sh smollm2-135m
#   ./scripts/download_model.sh llama-3.1-8b
#   ./scripts/download_model.sh mistral-7b
#   ./scripts/download_model.sh mixtral-8x7b
#   ./scripts/download_model.sh phi-3-mini
#   ./scripts/download_model.sh gemma-2b
#
# Models are downloaded to ~/.obelyzk/models/<model-name>/
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

MODEL_DIR="${OBELYZK_MODEL_DIR:-$HOME/.obelyzk/models}"
MODEL="${1:-}"

if [ -z "$MODEL" ]; then
    echo "ObelyZK Model Downloader"
    echo ""
    echo "Usage: $0 <model-name>"
    echo ""
    echo "Supported models (full ZK proof — every operation cryptographically proven):"
    echo ""
    echo "  Model               Params   Architecture         Size"
    echo "  ─────────────────── ──────── ──────────────────── ─────"
    echo "  qwen2.5-14b         14B      Qwen2 (GQA)          30 GB  ← verified on Starknet"
    echo "  qwen2.5-7b          7B       Qwen2 (GQA)          15 GB"
    echo "  glm-4-9b            9B       ChatGLM (fused QKV)   18 GB"
    echo "  minimax-m2.5        256B MoE MiniMax (256 experts) 400 GB (FP8)"
    echo "  kimi-k2.5           1T MoE   DeepSeek-V3 (MLA)    ~600 GB"
    echo "  llama-3.1-8b        8B       LLaMA (GQA)          16 GB"
    echo "  mistral-7b          7B       Mistral (GQA+SWA)     15 GB"
    echo "  mixtral-8x7b        47B MoE  Mixtral (8 experts)   87 GB"
    echo "  phi-3-mini          3.8B     Phi (fused QKV)        8 GB"
    echo "  gemma-2b            2B       Gemma                  5 GB"
    echo "  smollm2-135m        135M     SmolLM                 0.3 GB  ← fastest for testing"
    echo ""
    echo "Models are saved to: $MODEL_DIR/<model-name>/"
    echo ""
    echo "After downloading, prove with:"
    echo "  echo 'Hello' | OBELYSK_MODEL_DIR=$MODEL_DIR/<model-name> obelyzk chat --model local"
    exit 1
fi

# Check for huggingface-cli
if ! command -v huggingface-cli &>/dev/null; then
    echo "Installing huggingface_hub..."
    pip install -q huggingface_hub
fi

download_hf() {
    local repo="$1"
    local dest="$2"
    echo "Downloading $repo → $dest"
    echo "This may take a while for large models..."
    huggingface-cli download "$repo" --local-dir "$dest"
    echo "Done: $(du -sh "$dest" | cut -f1) in $dest"
}

download_small() {
    local repo="$1"
    local dest="$2"
    mkdir -p "$dest"
    echo "Downloading $repo → $dest"
    for f in config.json tokenizer.json tokenizer_config.json; do
        curl -sL "https://huggingface.co/$repo/resolve/main/$f" -o "$dest/$f" 2>/dev/null || true
    done
    # SafeTensors — might be single file or sharded
    if curl -sIf "https://huggingface.co/$repo/resolve/main/model.safetensors" >/dev/null 2>&1; then
        curl -L "https://huggingface.co/$repo/resolve/main/model.safetensors" -o "$dest/model.safetensors"
    else
        huggingface-cli download "$repo" --local-dir "$dest"
    fi
    echo "Done: $(du -sh "$dest" | cut -f1) in $dest"
}

DEST="$MODEL_DIR/$MODEL"

case "$MODEL" in
    qwen2.5-14b|qwen-14b)
        download_hf "Qwen/Qwen2.5-14B" "$DEST"
        ;;
    qwen2.5-7b|qwen-7b)
        download_hf "Qwen/Qwen2.5-7B" "$DEST"
        ;;
    glm-4-9b|glm4|chatglm)
        download_hf "THUDM/glm-4-9b" "$DEST"
        ;;
    minimax-m2.5|minimax)
        echo "WARNING: MiniMax-M2.5 is ~400 GB (FP8 quantized, 256 experts)."
        echo "         FP8 dequantization is supported in ObelyZK v0.4.0+."
        echo ""
        read -p "Continue? [y/N] " -n 1 -r
        echo
        [[ $REPLY =~ ^[Yy]$ ]] || exit 0
        download_hf "MiniMaxAI/MiniMax-M2.5" "$DEST"
        ;;
    kimi-k2.5|kimi)
        echo "WARNING: Kimi-K2.5 is ~600 GB and uses MLA attention."
        echo "         MLA attention path is in development."
        echo ""
        read -p "Continue? [y/N] " -n 1 -r
        echo
        [[ $REPLY =~ ^[Yy]$ ]] || exit 0
        download_hf "moonshotai/Kimi-K2.5" "$DEST"
        ;;
    llama-3.1-8b|llama-8b|llama)
        download_hf "meta-llama/Llama-3.1-8B" "$DEST"
        ;;
    mistral-7b|mistral)
        download_hf "mistralai/Mistral-7B-v0.3" "$DEST"
        ;;
    mixtral-8x7b|mixtral)
        download_hf "mistralai/Mixtral-8x7B-v0.1" "$DEST"
        ;;
    phi-3-mini|phi3|phi)
        download_hf "microsoft/Phi-3-mini-4k-instruct" "$DEST"
        ;;
    gemma-2b|gemma)
        download_hf "google/gemma-2b" "$DEST"
        ;;
    smollm2-135m|smollm|smol)
        download_small "HuggingFaceTB/SmolLM2-135M" "$DEST"
        ;;
    *)
        echo "Unknown model: $MODEL"
        echo "Run '$0' without arguments to see supported models."
        exit 1
        ;;
esac

echo ""
echo "To prove with this model:"
echo "  echo 'Hello' | OBELYSK_MODEL_DIR=$DEST obelyzk chat --model local"
