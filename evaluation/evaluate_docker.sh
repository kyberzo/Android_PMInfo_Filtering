#!/bin/bash
# Evaluate models in Docker containers
# This script evaluates each model in isolated Docker containers to avoid TensorFlow compatibility issues

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT"  # Test data is in project root
MODELS_DIR="$PROJECT_ROOT/models/output"
OUTPUT_DIR="$SCRIPT_DIR/results"
IMAGE_NAME="tf-model-evaluator"
TAG="latest"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           Docker-Based Model Evaluation Framework                              ║${NC}"
echo -e "${BLUE}║  Evaluates each model in isolated containers to eliminate TF compatibility     ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════════════════╝${NC}"
echo

# Parse arguments
MODELS_TO_EVAL=()
BUILD_IMAGE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODELS_TO_EVAL+=("$2")
            shift 2
            ;;
        --all)
            MODELS_TO_EVAL=(bilstm transformer cnn cnn_lstm dummy features xgboost)
            shift
            ;;
        --build)
            BUILD_IMAGE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model <name>    Evaluate specific model (bilstm, cnn, dummy, etc.)"
            echo "  --all             Evaluate all models"
            echo "  --build           Rebuild Docker image before evaluation"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# If no models specified, show usage
if [ ${#MODELS_TO_EVAL[@]} -eq 0 ]; then
    echo -e "${YELLOW}No models specified.${NC}"
    echo "Usage: $0 --model <name> [--model <name2> ...]"
    echo "   or: $0 --all"
    echo ""
    echo "Available models:"
    echo "  - bilstm       : Bidirectional LSTM"
    echo "  - transformer  : Transformer architecture"
    echo "  - cnn          : 1D Convolutional Neural Network"
    echo "  - cnn_lstm     : CNN + LSTM Hybrid"
    echo "  - dummy        : Baseline LSTM"
    echo "  - features     : Feature-Enhanced LSTM with engineered features"
    echo "  - xgboost      : XGBoost gradient boosting on engineered features"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo -e "${GREEN}✓ Output directory: $OUTPUT_DIR${NC}"
echo

# Build Docker image if needed
if [ "$BUILD_IMAGE" = true ] || [ ! "$(docker images -q $IMAGE_NAME:$TAG 2>/dev/null)" ]; then
    echo -e "${BLUE}Building Docker image: $IMAGE_NAME:$TAG${NC}"
    docker build -t "$IMAGE_NAME:$TAG" -f "$SCRIPT_DIR/Dockerfile.evaluation" "$SCRIPT_DIR"
    echo -e "${GREEN}✓ Docker image built${NC}"
    echo
fi

# Evaluate each model
RESULTS=()
for model_name in "${MODELS_TO_EVAL[@]}"; do
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Evaluating: $model_name${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo

    # Run Docker container
    docker run --rm \
        -v "$DATA_DIR:/workspace/data" \
        -v "$MODELS_DIR:/workspace/models" \
        -v "$OUTPUT_DIR:/workspace/output" \
        "$IMAGE_NAME:$TAG" \
        python3 evaluate_model_in_docker.py \
            --model "$model_name" \
            --output /workspace/output

    RESULT_FILE="$OUTPUT_DIR/${model_name}_evaluation_result.json"
    if [ -f "$RESULT_FILE" ]; then
        STATUS=$(jq -r '.status' "$RESULT_FILE")
        ACCURACY=$(jq -r '.accuracy // "N/A"' "$RESULT_FILE")

        if [ "$STATUS" = "SUCCESS" ]; then
            echo -e "${GREEN}✓ $model_name evaluated successfully${NC}"
            echo -e "${GREEN}  Accuracy: $ACCURACY${NC}"
            RESULTS+=("$model_name: $ACCURACY")
        else
            ERROR=$(jq -r '.error // "Unknown error"' "$RESULT_FILE")
            echo -e "${RED}✗ $model_name evaluation failed${NC}"
            echo -e "${RED}  Error: $ERROR${NC}"
        fi
    else
        echo -e "${RED}✗ No result file found for $model_name${NC}"
    fi
    echo
done

# Summary
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                           EVALUATION SUMMARY                                  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════════════════╝${NC}"
echo

if [ ${#RESULTS[@]} -eq 0 ]; then
    echo -e "${RED}No models were successfully evaluated${NC}"
else
    echo -e "${GREEN}Successfully evaluated models:${NC}"
    for result in "${RESULTS[@]}"; do
        echo -e "  ${GREEN}✓${NC} $result"
    done
fi

echo
echo -e "${GREEN}Results saved to: $OUTPUT_DIR${NC}"
echo

# Create summary report
echo -e "${BLUE}Generating summary report...${NC}"
python3 "$SCRIPT_DIR/generate_evaluation_summary.py" "$OUTPUT_DIR"

echo -e "${GREEN}✓ Evaluation complete${NC}"
