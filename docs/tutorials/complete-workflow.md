# Complete CLI Workflow Tutorial

This tutorial demonstrates a complete machine learning workflow using ModelGardener CLI commands, from project creation to model deployment.

## Overview

ModelGardener provides a comprehensive CLI interface for the entire machine learning pipeline:

1. **Project Setup**: Create and configure new ML projects
2. **System Validation**: Check environment and dependencies
3. **Model Training**: Train models with advanced configurations
4. **Model Evaluation**: Comprehensive performance analysis
5. **Prediction**: Generate predictions on new data
6. **Deployment**: Deploy models to various formats and platforms

## Prerequisites

Before starting, ensure ModelGardener is properly installed and configured:

```bash
# Verify installation
modelgardener_cli.py check --all

# Check available models
modelgardener_cli.py models --list
```

## Complete Workflow Example

### Step 1: Project Creation

Create a new image classification project with interactive setup:

```bash
# Create project with interactive configuration
modelgardener_cli.py create image_classifier \
    --dir ./projects/image_classifier \
    --interactive \
    --model-family efficientnet \
    --model-name EfficientNet-B3 \
    --num-classes 10 \
    --epochs 100

# Navigate to project directory
cd ./projects/image_classifier
```

**Generated Project Structure:**
```
image_classifier/
├── config.yaml
├── train.py
├── evaluation.py
├── prediction.py
├── deploy.py
├── data/
│   ├── train/
│   └── val/
├── custom_modules/
└── logs/
```

### Step 2: Configuration Validation

Validate and optimize the project configuration:

```bash
# Validate configuration with optimization suggestions
modelgardener_cli.py config \
    --file config.yaml \
    --validate \
    --strict \
    --suggest-optimizations

# Interactive configuration editing if needed
modelgardener_cli.py config \
    --file config.yaml \
    --edit
```

### Step 3: System Health Check

Perform comprehensive system validation before training:

```bash
# Complete system and project check
modelgardener_cli.py check \
    --all \
    --project . \
    --performance \
    --format html \
    --output pre_training_check.html

# Fix any detected issues automatically
modelgardener_cli.py check \
    --all \
    --fix
```

### Step 4: Data Preparation

Ensure your training data is properly organized:

```
data/
├── train/
│   ├── class_0/
│   │   ├── image_001.jpg
│   │   └── ...
│   ├── class_1/
│   └── ...
└── val/
    ├── class_0/
    └── ...
```

### Step 5: Model Training

Train the model with comprehensive monitoring:

```bash
# Start training with GPU acceleration and mixed precision
modelgardener_cli.py train \
    --config config.yaml \
    --gpus 2 \
    --mixed-precision \
    --tensorboard \
    --early-stopping

# Monitor training progress
# TensorBoard: http://localhost:6006
```

**Training Features:**
- Automatic checkpointing
- Early stopping
- Learning rate scheduling
- Mixed precision training
- Multi-GPU support
- Real-time monitoring

### Step 6: Model Evaluation

Comprehensive model evaluation with detailed analysis:

```bash
# Detailed evaluation with all features
modelgardener_cli.py evaluate \
    --config config.yaml \
    --model ./logs/models/best_model.keras \
    --per-class \
    --confusion-matrix \
    --roc-curves \
    --precision-recall \
    --interpretability \
    --grad-cam \
    --detailed-report \
    --format html,json \
    --output-dir ./evaluation_results
```

**Evaluation Outputs:**
- Comprehensive metrics analysis
- Confusion matrices and ROC curves
- Per-class performance breakdown
- Model interpretability visualizations
- Interactive HTML report

### Step 7: Model Prediction

Test the model with new data:

```bash
# Single image prediction with visualization
modelgardener_cli.py predict \
    --config config.yaml \
    --model ./logs/models/best_model.keras \
    --input test_image.jpg \
    --visualize \
    --grad-cam \
    --top-k 5 \
    --detailed-output

# Batch prediction on directory
modelgardener_cli.py predict \
    --config config.yaml \
    --model ./logs/models/best_model.keras \
    --input ./test_images/ \
    --batch-size 32 \
    --output-format csv \
    --benchmark
```

### Step 8: Model Deployment

Deploy the model to multiple formats:

```bash
# Multi-format deployment with optimization
modelgardener_cli.py deploy \
    --config config.yaml \
    --model ./logs/models/best_model.keras \
    --format all \
    --optimize \
    --quantize int8 \
    --target-platform cpu \
    --encrypt \
    --serve \
    --docker \
    --port 8080

# Test the deployed API
curl -X POST \
    -F "image=@test_image.jpg" \
    http://localhost:8080/predict
```

## Advanced Workflows

### Cross-Validation Training

```bash
# 5-fold cross-validation with hyperparameter tuning
modelgardener_cli.py train \
    --config config.yaml \
    --cross-validation 5 \
    --hyperparameter-tuning \
    --epochs 200

# Evaluate cross-validation results
modelgardener_cli.py evaluate \
    --config config.yaml \
    --model ./logs/cross_validation/best_fold_model.keras \
    --detailed-report
```

### Transfer Learning Workflow

```bash
# Create transfer learning project
modelgardener_cli.py create transfer_learning_project \
    --model-family efficientnet \
    --model-name EfficientNet-B0 \
    --num-classes 3 \
    --epochs 50

# Fine-tune with frozen base layers
modelgardener_cli.py train \
    --config config.yaml \
    --fine-tune \
    --pretrained \
    --learning-rate 0.0001 \
    --freeze-base-layers
```

### Production Deployment Workflow

```bash
# Comprehensive production deployment
modelgardener_cli.py deploy \
    --config config.yaml \
    --model ./logs/models/production_model.keras \
    --format onnx,tflite \
    --optimize \
    --quantize int8 \
    --encrypt \
    --secure-serving \
    --docker \
    --package \
    --documentation

# Performance validation
modelgardener_cli.py predict \
    --model ./deployment/models/model.onnx \
    --input ./validation_set/ \
    --benchmark \
    --output-format json
```

## Model Comparison Workflow

### Multiple Model Training

```bash
# Train multiple model architectures
for model in "ResNet-50" "EfficientNet-B0" "MobileNet-V2"; do
    # Create configuration for each model
    modelgardener_cli.py config \
        --template advanced \
        --output config_${model}.yaml
    
    # Update model selection in config
    sed -i "s/selected_model_name:.*/selected_model_name: \"${model}\"/" config_${model}.yaml
    
    # Train model
    modelgardener_cli.py train \
        --config config_${model}.yaml \
        --output-dir ./experiments/${model}/
done
```

### Model Performance Comparison

```bash
# Evaluate all models
for model in "ResNet-50" "EfficientNet-B0" "MobileNet-V2"; do
    modelgardener_cli.py evaluate \
        --config config_${model}.yaml \
        --model ./experiments/${model}/models/best_model.keras \
        --output-dir ./evaluations/${model}/ \
        --format json
done

# Compare performance metrics
python scripts/compare_models.py \
    --results ./evaluations/*/metrics/overall_metrics.json \
    --output model_comparison.html
```

## Custom Function Integration

### Custom Model Architecture

```python
# custom_modules/custom_models.py
def create_attention_cnn(input_shape, num_classes, **kwargs):
    """Custom CNN with attention mechanism."""
    # Implementation here
    pass
```

```bash
# Register and use custom model
modelgardener_cli.py models \
    --register ./custom_modules/custom_models.py

# Update configuration to use custom model
modelgardener_cli.py config \
    --file config.yaml \
    --edit  # Select custom model family
```

### Custom Training Loop

```python
# custom_modules/custom_training_loops.py
def custom_training_with_adversarial(model, train_data, val_data, **kwargs):
    """Custom training loop with adversarial training."""
    # Implementation here
    pass
```

## Automated Workflows

### CI/CD Pipeline Integration

```bash
#!/bin/bash
# ci_ml_pipeline.sh

# Environment validation
modelgardener_cli.py check \
    --environment \
    --critical-only \
    --format json \
    --output ci_check.json

if [ $? -ne 0 ]; then
    echo "Environment check failed"
    exit 1
fi

# Model training
modelgardener_cli.py train \
    --config production_config.yaml \
    --epochs 100

# Model validation
modelgardener_cli.py evaluate \
    --config production_config.yaml \
    --model ./logs/models/best_model.keras \
    --format json \
    --output evaluation_results.json

# Performance threshold check
python scripts/validate_performance.py \
    --results evaluation_results.json \
    --min-accuracy 0.85

# Deployment if validation passes
if [ $? -eq 0 ]; then
    modelgardener_cli.py deploy \
        --config production_config.yaml \
        --model ./logs/models/best_model.keras \
        --format onnx \
        --optimize \
        --package
fi
```

### Batch Processing Pipeline

```bash
#!/bin/bash
# batch_processing.sh

# Process multiple datasets
for dataset in dataset1 dataset2 dataset3; do
    echo "Processing ${dataset}..."
    
    # Create project
    modelgardener_cli.py create ${dataset}_project \
        --dir ./projects/${dataset} \
        --template advanced
    
    cd ./projects/${dataset}
    
    # Link dataset
    ln -s ../../datasets/${dataset}/train ./data/train
    ln -s ../../datasets/${dataset}/val ./data/val
    
    # Train model
    modelgardener_cli.py train \
        --config config.yaml \
        --epochs 50
    
    # Evaluate model
    modelgardener_cli.py evaluate \
        --config config.yaml \
        --model ./logs/models/best_model.keras \
        --output-dir ../results/${dataset}/
    
    cd ../..
done

# Generate comparison report
python scripts/batch_comparison.py \
    --results ./projects/results/*/metrics/ \
    --output batch_comparison_report.html
```

## Monitoring and Maintenance

### Regular Health Monitoring

```bash
#!/bin/bash
# health_monitor.sh

# Daily health check
modelgardener_cli.py check \
    --all \
    --format json \
    --output /var/log/modelgardener/health_$(date +%Y%m%d).json

# Performance tracking
modelgardener_cli.py check \
    --performance \
    --benchmark \
    --output /var/log/modelgardener/performance_$(date +%Y%m%d).json

# Alert on critical issues
python scripts/health_alerting.py \
    --health-file /var/log/modelgardener/health_$(date +%Y%m%d).json \
    --alert-webhook https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

### Model Performance Monitoring

```bash
# Monitor deployed model performance
modelgardener_cli.py predict \
    --model production_model.onnx \
    --input /data/daily_samples/ \
    --benchmark \
    --output daily_performance.json

# Check for model drift
python scripts/drift_detection.py \
    --predictions daily_performance.json \
    --baseline baseline_performance.json \
    --threshold 0.05
```

## Best Practices Summary

### Project Organization

1. **Structured Workflows**: Use consistent project structures
2. **Configuration Management**: Version control all configurations
3. **Automated Validation**: Include health checks in workflows
4. **Documentation**: Maintain comprehensive project documentation

### Performance Optimization

1. **Resource Monitoring**: Regular system health checks
2. **Training Optimization**: Use mixed precision and multi-GPU
3. **Model Selection**: Choose appropriate architectures for tasks
4. **Deployment Optimization**: Optimize models for target platforms

### Production Readiness

1. **Comprehensive Testing**: Evaluate models thoroughly before deployment
2. **Security**: Use encryption and secure serving for production
3. **Monitoring**: Implement continuous performance monitoring
4. **Maintenance**: Regular health checks and model updates

## Troubleshooting Common Issues

### Training Issues

```bash
# Memory issues
modelgardener_cli.py check --performance --memory-usage
# Solution: Reduce batch size or enable mixed precision

# GPU not detected
modelgardener_cli.py check --hardware --gpu
# Solution: Check CUDA installation and drivers

# Slow training
modelgardener_cli.py check --performance --benchmark
# Solution: Optimize data pipeline or use distributed training
```

### Deployment Issues

```bash
# Model conversion errors
modelgardener_cli.py models --validate ./models/model.keras
# Solution: Check model compatibility with target format

# API server issues
modelgardener_cli.py deploy --serve --test-endpoints
# Solution: Check port availability and permissions
```

### Environment Issues

```bash
# Dependency conflicts
modelgardener_cli.py check --dependencies --verbose
# Solution: Create fresh virtual environment

# Permission errors
modelgardener_cli.py check --permissions --fix
# Solution: Adjust file and directory permissions
```

## Next Steps

After completing this tutorial, consider exploring:

1. **Advanced Custom Functions**: Develop custom models, loss functions, and metrics
2. **Distributed Training**: Scale training across multiple machines
3. **MLOps Integration**: Integrate with MLflow, Kubeflow, or similar platforms
4. **Edge Deployment**: Deploy models to mobile and IoT devices
5. **Continuous Learning**: Implement online learning and model updates

## See Also

- [CLI Command Reference](README.md)
- [Configuration Guide](configuration.md)
- [Custom Functions Tutorial](custom-functions.md)
- [Deployment Guide](deployment.md)
- [Performance Optimization](performance-optimization.md)
