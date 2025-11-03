#!/bin/bash
# Train models using Docker
# Usage: ./train_docker.sh [model] [options]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
EPOCHS=${EPOCHS:-1000}
BATCH_SIZE=${BATCH_SIZE:-1024}
PATIENCE=${PATIENCE:-8}

print_usage() {
    echo "Usage: $0 [model] [options]"
    echo ""
    echo "Models:"
    echo "  bilstm       - Train Bidirectional LSTM"
    echo "  transformer  - Train Transformer"
    echo "  cnn          - Train 1D CNN"
    echo "  cnn_lstm     - Train CNN+LSTM Hybrid"
    echo "  dummy        - Train Dummy LSTM (benchmark baseline)"
    echo "  features     - Train Multi-input LSTM with 21 engineered features"
    echo "  xgboost      - Train XGBoost Gradient Boosting"
    echo "  all          - Train all models in parallel"
    echo ""
    echo "Options:"
    echo "  --epochs N        - Number of training epochs (default: 100)"
    echo "  --batch-size N    - Batch size (default: 1024, 512 for features)"
    echo "  --patience N      - Early stopping patience (default: 8)"
    echo "  --rebuild         - Rebuild Docker images before training"
    echo "  --no-cache        - Build without cache"
    echo ""
    echo "Examples:"
    echo "  $0 bilstm"
    echo "  $0 all --epochs 50 --batch-size 512"
    echo "  $0 dummy --epochs 100  # Train dummy benchmark model"
    echo "  $0 features --epochs 100  # Train features model"
    echo "  $0 all --epochs 5 --patience 2  # Quick test"
}

# Parse arguments
MODEL=""
REBUILD=false
NO_CACHE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        bilstm|transformer|cnn|cnn_lstm|dummy|features|xgboost|all)
            MODEL="$1"
            shift
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --patience)
            PATIENCE="$2"
            shift 2
            ;;
        --rebuild)
            REBUILD=true
            shift
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

if [ -z "$MODEL" ]; then
    echo -e "${RED}Error: Model not specified${NC}"
    print_usage
    exit 1
fi

# Export environment variables for docker-compose
export EPOCHS
export BATCH_SIZE
export PATIENCE

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}Docker Training - Android Package Name Classifier${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  Model(s): $MODEL"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Early stopping patience: $PATIENCE"
echo ""

# Create output directories
mkdir -p output/{bilstm,transformer,cnn,cnn_lstm,dummy,features,xgboost}

# Rebuild if requested
if [ "$REBUILD" = true ]; then
    echo -e "${YELLOW}Rebuilding Docker images...${NC}"
    if [ "$MODEL" = "all" ]; then
        docker-compose build $NO_CACHE bilstm transformer cnn cnn_lstm dummy features xgboost
    else
        docker-compose build $NO_CACHE $MODEL
    fi
    echo ""
fi

# Start training
echo -e "${GREEN}Starting training...${NC}"
echo ""

START_TIME=$(date +%s)

if [ "$MODEL" = "all" ]; then
    echo -e "${YELLOW}Training all models in parallel...${NC}"
    docker-compose --profile all up --abort-on-container-exit
else
    echo -e "${YELLOW}Training $MODEL model...${NC}"
    docker-compose --profile $MODEL up --abort-on-container-exit
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo -e "${BLUE}======================================================================${NC}"
echo -e "${GREEN}Training completed!${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""
echo "Total time: $((DURATION / 60)) minutes $((DURATION % 60)) seconds"
echo ""
echo -e "${GREEN}Output locations:${NC}"

if [ "$MODEL" = "all" ]; then
    for m in bilstm transformer cnn cnn_lstm dummy features xgboost; do
        if [ -d "output/$m" ]; then
            echo "  $m: output/$m/"
        fi
    done
else
    echo "  output/$MODEL/"
fi

echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Review training results in output directories"
echo "  2. Compare models: python3 ../deploy.py"
echo "  3. Deploy best model to production"
echo ""
