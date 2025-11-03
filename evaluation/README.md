# Docker-Based Model Evaluation Framework

This directory contains a Docker-based evaluation infrastructure that eliminates TensorFlow compatibility issues by evaluating each model in isolated containers.

**Problem Solved:**
- TensorFlow 1.x models won't load in TensorFlow 2.x environment
- Different model architectures may have version-specific requirements
- Debugging compatibility issues across multiple models is difficult

**Solution:**
- Each model runs in its own Docker container with appropriate TensorFlow version
- Clean separation of concerns
- Reproducible evaluation across different machines
- Easy to add support for new TensorFlow versions or model types

## Quick Start

### Evaluate All Models

```bash
cd evaluation
./evaluate_docker.sh --all
```

### Evaluate Specific Models

```bash
# Single model
./evaluate_docker.sh --model bilstm

# Multiple specific models
./evaluate_docker.sh --model bilstm --model transformer --model cnn
```

### Build Docker Image (if needed)

```bash
./evaluate_docker.sh --all --build
```

## Directory Structure

```
evaluation/
├── README.md                          ← You are here
├── Dockerfile.evaluation              ← Docker image definition
├── evaluate_all_models.py             ← Full evaluation script (for comparison)
├── evaluate_model_in_docker.py        ← Single model evaluation (runs in Docker)
├── evaluate_docker.sh                 ← Main orchestrator script
├── generate_evaluation_summary.py     ← Summary report generator
└── results/                           ← Evaluation results (created after running)
    ├── bilstm_evaluation_result.json
    ├── cnn_evaluation_result.json
    ├── ...
    ├── EVALUATION_SUMMARY.md          ← Human-readable report
    └── evaluation_summary.json        ← Machine-readable report
```

## How It Works

### 1. Build Docker Image

The Dockerfile installs TensorFlow 2.8 and all required dependencies:
- TensorFlow 2.8.0
- Keras 2.8.0
- XGBoost
- scikit-learn
- h5py

### 2. Run Evaluation

For each model, the script:
1. Starts a Docker container with the evaluation image
2. Mounts data and model directories
3. Runs the model evaluation inside the container
4. Saves results to the `results/` directory

### 3. Generate Summary

After all models are evaluated, generates:
- `EVALUATION_SUMMARY.md` - Human-readable report
- `evaluation_summary.json` - Machine-readable report
- Individual JSON results for each model

## Supported Models

| Model | Status | Notes |
|-------|--------|-------|
| bilstm | ✓ Ready | Bidirectional LSTM |
| transformer | ✓ Ready | Transformer architecture |
| cnn | ✓ Ready | 1D Convolutional Neural Network |
| cnn_lstm | ✓ Ready | CNN + LSTM Hybrid |
| dummy | ✓ Ready | Baseline LSTM |
| features | ✓ Ready | Multi-input LSTM with features |

## Data Requirements

The evaluation expects the following file structure:

```
project_root/
├── 0_test.csv              ← Legitimate packages (one per line)
├── 1_test.csv              ← Suspicious packages (one per line)
├── models/
│   └── output/
│       ├── bilstm/
│       │   ├── bilstm_model_*.hdf5
│       │   └── bilstm_mlinfo_*.json
│       ├── cnn/
│       │   ├── cnn_model_*.hdf5
│       │   └── cnn_mlinfo_*.json
│       └── ... (other models)
└── evaluation/             ← This directory
```

## Output Files

Each model produces a JSON file with structure:

```json
{
  "model_name": "bilstm",
  "model_path": "/path/to/model.hdf5",
  "status": "SUCCESS",
  "accuracy": 0.8739,
  "precision": 0.9484,
  "recall": 0.7907,
  "f1_score": 0.8624,
  "roc_auc": 0.9430,
  "true_negatives": 20007,
  "false_positives": 899,
  "false_negatives": 4375,
  "true_positives": 16531,
  "model_size_mb": 18.5,
  "timestamp": "2025-10-23T19:30:00.123456"
}
```

## Docker Commands Reference

### Build Image

```bash
docker build -t tf-model-evaluator:latest -f Dockerfile.evaluation .
```

### Run Single Model Evaluation

```bash
docker run --rm \
  -v /path/to/data:/workspace/data \
  -v /path/to/models:/workspace/models \
  -v /path/to/results:/workspace/output \
  tf-model-evaluator:latest \
  python3 evaluate_model_in_docker.py --model bilstm --output /workspace/output
```

### Interactive Container

```bash
docker run -it --rm \
  -v /path/to/data:/workspace/data \
  -v /path/to/models:/workspace/models \
  tf-model-evaluator:latest \
  /bin/bash
```

## Advanced Usage

### Custom TensorFlow Version

Edit `Dockerfile.evaluation` to use a different base image:

```dockerfile
# Change this line
FROM tensorflow/tensorflow:2.8.0

# To any other version, e.g.
FROM tensorflow/tensorflow:2.12.0
```

Then rebuild:

```bash
docker build --no-cache -t tf-model-evaluator:latest -f Dockerfile.evaluation .
```

### Add Custom Layers

If a model uses custom layers, add them to `Dockerfile.evaluation`:

```dockerfile
# After RUN pip install...
COPY custom_layers.py /workspace/
RUN pip install -e /workspace/
```

### Debug Container

Run a container with interactive shell:

```bash
./evaluate_docker.sh --model bilstm --build  # First time only

# Then start interactive container
docker run -it --rm \
  -v $PWD/../data:/workspace/data \
  -v $PWD/../models/output:/workspace/models \
  tf-model-evaluator:latest \
  /bin/bash
```

## Troubleshooting

### "Docker daemon not running"

Start Docker:
```bash
sudo systemctl start docker
```

### "Permission denied" error

Add user to docker group:
```bash
sudo usermod -aG docker $USER
newgrp docker
```

### Model file not found

Ensure the model directory structure matches expected format:
```bash
ls -la models/output/<model_name>/
# Should show: <model_name>_model_*.hdf5 and <model_name>_mlinfo_*.json
```

### Out of disk space

Clean up old Docker images:
```bash
docker system prune -a
```

## Performance Considerations

### Single Model Evaluation Time

On CPU (typical):
- CNN: 30-60 seconds
- LSTM/BiLSTM: 1-2 minutes
- Transformer: 2-5 minutes

Times include:
- Docker container startup: ~5-10 seconds
- Data loading: ~5-10 seconds
- Model loading: ~5-15 seconds
- Prediction: ~10-30 seconds (depends on model complexity)

### Parallel Evaluation

Currently evaluates models sequentially. To evaluate in parallel:

```bash
# Run multiple evaluation scripts in background
./evaluate_docker.sh --model bilstm &
./evaluate_docker.sh --model transformer &
./evaluate_docker.sh --model cnn &
wait
```

