# Docker-Based Training Guide

This guide explains how to train models using Docker containers for isolation, reproducibility, and parallel execution.

## Quick Start

### 1. Build Docker Images

```bash
cd models

# Build all images
./build_images.sh

# Build without cache (clean rebuild)
./build_images.sh --no-cache
```

This creates 5 Docker images:
- `tf-pname-bilstm` - Bidirectional LSTM
- `tf-pname-transformer` - Transformer model
- `tf-pname-cnn` - 1D CNN
- `tf-pname-cnn-lstm` - CNN+LSTM Hybrid
- `tf-pname-baseline` - Baseline LSTM

### 2. Train Models

```bash
# Train single model
./train_docker.sh bilstm

# Train all models in parallel
./train_docker.sh all

# Custom training parameters
./train_docker.sh transformer --epochs 50 --batch-size 512

# Quick test (5 epochs)
./train_docker.sh all --epochs 5 --patience 2
```

### 3. View Results

Training outputs are saved to `output/{model_name}/`:

```bash
ls -la output/bilstm/
# bilstm_model_20251016_143022.hdf5
# bilstm_mlinfo_20251016_143022.json
```

## Available Models

| Model | Container Name | Dockerfile | Expected Performance |
|-------|---------------|------------|---------------------|
| Baseline LSTM | `train-baseline` | [Dockerfile.baseline](Dockerfile.baseline) | Reference model |
| BiLSTM | `train-bilstm` | [Dockerfile.bilstm](Dockerfile.bilstm) | +2-3% accuracy |
| Transformer | `train-transformer` | [Dockerfile.transformer](Dockerfile.transformer) | +4-6% accuracy |
| 1D CNN | `train-cnn` | [Dockerfile.cnn](Dockerfile.cnn) | 10x faster inference |
| CNN+LSTM | `train-cnn-lstm` | [Dockerfile.cnn_lstm](Dockerfile.cnn_lstm) | +1-3% accuracy |

## Training Options

### Using train_docker.sh Script

```bash
./train_docker.sh [model] [options]

Models:
  bilstm       - Train Bidirectional LSTM
  transformer  - Train Transformer
  cnn          - Train 1D CNN
  cnn_lstm     - Train CNN+LSTM Hybrid
  baseline     - Train Baseline LSTM
  all          - Train all models in parallel

Options:
  --epochs N        - Number of training epochs (default: 100)
  --batch-size N    - Batch size (default: 1024)
  --patience N      - Early stopping patience (default: 8)
  --rebuild         - Rebuild Docker images before training
  --no-cache        - Build without cache
```

### Using docker-compose Directly

```bash
# Train specific model
docker-compose --profile bilstm up

# Train multiple models
docker-compose --profile bilstm --profile cnn up

# Train all models
docker-compose --profile all up

# Set environment variables
EPOCHS=50 BATCH_SIZE=512 docker-compose --profile transformer up
```

## Environment Variables

All models support these environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `EPOCHS` | 100 | Maximum training epochs |
| `BATCH_SIZE` | 1024 | Training batch size |
| `PATIENCE` | 8 | Early stopping patience |
| `EMBEDDING_DIM` | 128 | Embedding dimension |
| `DROPOUT` | 0.3-0.5 | Dropout rate |

### Model-Specific Variables

**BiLSTM:**
- `LSTM_UNITS` (default: 128)

**Transformer:**
- `D_MODEL` (default: 128)
- `NUM_HEADS` (default: 4)
- `FF_DIM` (default: 256)
- `NUM_BLOCKS` (default: 2)

**CNN:**
- `FILTERS` (default: 256)

**CNN+LSTM:**
- `CNN_FILTERS` (default: 64)
- `LSTM_UNITS` (default: 128)

## Examples

### Example 1: Quick Experiment

Train all models with reduced epochs for testing:

```bash
./train_docker.sh all --epochs 5 --patience 2
```

### Example 2: Production Training

Train BiLSTM with custom parameters:

```bash
./train_docker.sh bilstm \
  --epochs 200 \
  --batch-size 2048 \
  --patience 12 \
  --rebuild
```

### Example 3: Parallel Training with Different Settings

```bash
# Terminal 1: Train BiLSTM with dropout 0.3
DROPOUT=0.3 docker-compose --profile bilstm up

# Terminal 2: Train CNN with dropout 0.5
DROPOUT=0.5 docker-compose --profile cnn up

# Terminal 3: Train Transformer
docker-compose --profile transformer up
```

### Example 4: Cloud/GPU Training

```bash
# SSH to GPU instance
ssh gpu-server

# Pull repo and data
git clone <repo-url>
cd tf_android_package_name/models

# Build images on GPU machine
./build_images.sh

# Train with GPU support (modify Dockerfile to use tensorflow-gpu)
./train_docker.sh all --epochs 100
```

## Volume Mounts

Each container mounts:

1. **Data Volume (Read-Only):**
   - Host: `../` (parent directory with CSV files)
   - Container: `/workspace/data`
   - Contains: `0_train.csv`, `1_train.csv`, etc.

2. **Output Volume (Read-Write):**
   - Host: `./output/{model_name}/`
   - Container: `/workspace/output`
   - Contains: Trained models and mlinfo files

## Output Structure

```
models/
├── output/
│   ├── bilstm/
│   │   ├── bilstm_model_20251016_143022.hdf5
│   │   └── bilstm_mlinfo_20251016_143022.json
│   ├── transformer/
│   │   ├── transformer_model_20251016_143530.hdf5
│   │   └── transformer_mlinfo_20251016_143530.json
│   ├── cnn/
│   ├── cnn_lstm/
│   └── baseline/
```

## Resource Management

### Limit CPU and Memory

Edit `docker-compose.yml` to add resource limits:

```yaml
services:
  bilstm:
    # ... existing config ...
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G
```

### Monitor Resource Usage

```bash
# Watch all containers
docker stats

# Watch specific container
docker stats train-bilstm

# Check logs
docker logs -f train-bilstm
```

## Troubleshooting

### Container Exits Immediately

Check logs:
```bash
docker logs train-bilstm
```

Common issues:
- Missing data files (check volume mount)
- Invalid parameters
- Out of memory

### Data Not Found

Verify volume mount:
```bash
docker exec -it train-bilstm ls -la /workspace/data/
```

Should show CSV files: `0_train.csv`, `1_train.csv`, etc.

### Out of Memory

Reduce batch size:
```bash
./train_docker.sh bilstm --batch-size 512
```

### Rebuild After Code Changes

```bash
./train_docker.sh bilstm --rebuild
```

## Cleanup

### Remove Containers

```bash
# Using script
./clean_docker.sh --containers

# Manual
docker-compose down
```

### Remove Images

```bash
# Using script
./clean_docker.sh --images

# Manual
docker rmi tf-pname-bilstm tf-pname-transformer tf-pname-cnn tf-pname-cnn-lstm tf-pname-baseline
```

### Remove Training Outputs

```bash
# Using script
./clean_docker.sh --outputs

# Manual
rm -rf output/*
```

### Complete Cleanup

```bash
./clean_docker.sh --all
```

## Advanced Usage

### Custom Dockerfile

Create custom Dockerfile for experimentation:

```dockerfile
# Dockerfile.custom
FROM python:3.8-slim

RUN pip install tensorflow-gpu==2.8.0  # GPU support

COPY my_custom_model.py /workspace/
ENTRYPOINT ["python3", "my_custom_model.py"]
```

Add to `docker-compose.yml`:

```yaml
services:
  custom:
    build:
      context: .
      dockerfile: Dockerfile.custom
    volumes:
      - ..:/workspace/data:ro
      - ./output/custom:/workspace/output
    profiles:
      - custom
```

Train:
```bash
docker-compose --profile custom up
```

### Export Trained Model from Container

```bash
# Copy from stopped container
docker cp train-bilstm:/workspace/output/model.hdf5 ./

# Or use volume mount (already available in output/)
cp output/bilstm/bilstm_model_*.hdf5 ../tf_predict_pname/app/
```

### Run Interactive Training Session

```bash
# Start container in interactive mode
docker run -it --rm \
  -v $(pwd)/..:/workspace/data:ro \
  -v $(pwd)/output/bilstm:/workspace/output \
  tf-pname-bilstm:latest \
  bash

# Inside container
python3 train_bilstm.py --epochs 10 --batch-size 512
```

