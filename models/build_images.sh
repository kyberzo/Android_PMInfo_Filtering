#!/bin/bash
# Build all Docker images for model training

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}Building Docker Images for Model Training${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# Parse options
NO_CACHE=""
if [[ "$1" == "--no-cache" ]]; then
    NO_CACHE="--no-cache"
    echo -e "${YELLOW}Building without cache...${NC}"
    echo ""
fi

# Build each model image
MODELS=("bilstm" "transformer" "cnn" "cnn_lstm" "dummy" "features" "xgboost")

for model in "${MODELS[@]}"; do
    echo -e "${GREEN}Building $model image...${NC}"
    docker-compose build $NO_CACHE $model
    echo ""
done

echo -e "${BLUE}======================================================================${NC}"
echo -e "${GREEN}All images built successfully!${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# Show images
echo -e "${YELLOW}Created images:${NC}"
docker images | grep "tf-pname"
echo ""

echo -e "${YELLOW}Next steps:${NC}"
echo "  Train single model:  ./train_docker.sh bilstm"
echo "  Train all models:    ./train_docker.sh all"
echo "  Quick test:          ./train_docker.sh all --epochs 5 --patience 2"
echo ""
