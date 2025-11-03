#!/bin/bash
# Clean up Docker containers, images, and outputs

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --containers    - Stop and remove containers only"
    echo "  --images        - Remove Docker images only"
    echo "  --outputs       - Remove training outputs only"
    echo "  --all           - Remove everything (containers, images, outputs)"
    echo ""
    echo "Examples:"
    echo "  $0 --containers"
    echo "  $0 --all"
}

CLEAN_CONTAINERS=false
CLEAN_IMAGES=false
CLEAN_OUTPUTS=false

# Parse arguments
if [[ $# -eq 0 ]]; then
    print_usage
    exit 1
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --containers)
            CLEAN_CONTAINERS=true
            shift
            ;;
        --images)
            CLEAN_IMAGES=true
            shift
            ;;
        --outputs)
            CLEAN_OUTPUTS=true
            shift
            ;;
        --all)
            CLEAN_CONTAINERS=true
            CLEAN_IMAGES=true
            CLEAN_OUTPUTS=true
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

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}Docker Cleanup${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# Clean containers
if [ "$CLEAN_CONTAINERS" = true ]; then
    echo -e "${YELLOW}Stopping and removing containers...${NC}"
    docker-compose down --remove-orphans 2>/dev/null || true

    # Remove specific containers if they exist
    for container in train-bilstm train-transformer train-cnn train-cnn-lstm train-baseline; do
        if docker ps -a | grep -q $container; then
            docker rm -f $container 2>/dev/null || true
        fi
    done
    echo -e "${GREEN}✓ Containers removed${NC}"
    echo ""
fi

# Clean images
if [ "$CLEAN_IMAGES" = true ]; then
    echo -e "${YELLOW}Removing Docker images...${NC}"

    for image in tf-pname-bilstm tf-pname-transformer tf-pname-cnn tf-pname-cnn-lstm tf-pname-baseline; do
        if docker images | grep -q $image; then
            docker rmi -f $image:latest 2>/dev/null || true
        fi
    done

    echo -e "${GREEN}✓ Images removed${NC}"
    echo ""
fi

# Clean outputs
if [ "$CLEAN_OUTPUTS" = true ]; then
    echo -e "${YELLOW}Removing training outputs...${NC}"

    if [ -d "output" ]; then
        echo "  Removing: output/"
        rm -rf output/*
        echo -e "${GREEN}✓ Outputs removed${NC}"
    else
        echo "  No outputs to remove"
    fi
    echo ""
fi

echo -e "${BLUE}======================================================================${NC}"
echo -e "${GREEN}Cleanup completed!${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""
