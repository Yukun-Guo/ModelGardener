# `deploy` Command

Deploy trained models to multiple formats and platforms with support for optimization, quantization, encryption, and serving capabilities.

## Synopsis

```bash
modelgardener_cli.py deploy [OPTIONS]
```

## Description

The `deploy` command provides comprehensive model deployment capabilities including:

- Multi-format model conversion (ONNX, TensorFlow Lite, TensorFlow.js)
- Model optimization and quantization for edge deployment
- Model encryption and security features
- REST API server deployment
- Performance benchmarking and monitoring
- Docker containerization support
- Cloud platform integration

## Options

### Model and Configuration

| Option | Short | Type | Description | Default |
|--------|-------|------|-------------|---------|
| `--config` | `-c` | `str` | Path to YAML configuration file | `config.yaml` |
| `--model` | `-m` | `str` | Path to trained model file | From config |
| `--weights` | `-w` | `str` | Path to model weights file | None |

### Deployment Format

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--format` | `str` | Deployment format (onnx, tflite, tfjs, keras, all) | `keras` |
| `--optimize` | `flag` | Enable model optimization | True |
| `--quantize` | `str` | Quantization type (int8, fp16, dynamic) | None |
| `--compress` | `flag` | Enable model compression | False |

### Security Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--encrypt` | `flag` | Enable model encryption | False |
| `--encryption-key` | `str` | Encryption key file path | Auto-generate |
| `--secure-serving` | `flag` | Enable secure HTTPS serving | False |
| `--api-key` | `str` | API key for secured endpoints | Auto-generate |

### Serving Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--serve` | `flag` | Start REST API server | False |
| `--host` | `str` | Server host address | `0.0.0.0` |
| `--port` | `int` | Server port number | `8000` |
| `--workers` | `int` | Number of worker processes | Auto |

### Optimization Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--target-platform` | `str` | Target deployment platform (cpu, gpu, edge, mobile) | `cpu` |
| `--batch-size` | `int` | Optimization batch size | 1 |
| `--input-shape` | `str` | Input shape for optimization | From model |
| `--precision` | `str` | Model precision (fp32, fp16, int8) | `fp32` |

### Output Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--output-dir` | `str` | Directory for deployment outputs | `./deployment` |
| `--package` | `flag` | Create deployment package | False |
| `--docker` | `flag` | Generate Docker configuration | False |
| `--documentation` | `flag` | Generate API documentation | True |

## Usage Examples

### Basic Model Deployment

```bash
# Deploy model to ONNX format
modelgardener_cli.py deploy --format onnx

# Deploy with optimization
modelgardener_cli.py deploy --format onnx --optimize --quantize int8

# Deploy to multiple formats
modelgardener_cli.py deploy --format all --optimize
```

### Edge Deployment

```bash
# Mobile/Edge optimized deployment
modelgardener_cli.py deploy \
    --format tflite \
    --target-platform mobile \
    --quantize int8 \
    --optimize \
    --compress

# Raspberry Pi deployment
modelgardener_cli.py deploy \
    --format tflite \
    --target-platform edge \
    --quantize int8 \
    --batch-size 1
```

### Secure Deployment

```bash
# Encrypted model deployment
modelgardener_cli.py deploy \
    --format onnx \
    --encrypt \
    --encryption-key ./keys/model.key

# Secure API deployment
modelgardener_cli.py deploy \
    --serve \
    --secure-serving \
    --api-key ./keys/api.key \
    --encrypt
```

### Server Deployment

```bash
# REST API server
modelgardener_cli.py deploy \
    --serve \
    --host 0.0.0.0 \
    --port 8080 \
    --workers 4

# Production server with security
modelgardener_cli.py deploy \
    --serve \
    --secure-serving \
    --host 0.0.0.0 \
    --port 443 \
    --workers 8 \
    --api-key production.key
```

### Containerized Deployment

```bash
# Docker deployment
modelgardener_cli.py deploy \
    --format onnx \
    --docker \
    --package \
    --serve

# Cloud-ready deployment
modelgardener_cli.py deploy \
    --format all \
    --docker \
    --secure-serving \
    --package \
    --documentation
```

## Deployment Formats

### ONNX Deployment

**Features:**
- Cross-platform compatibility
- Hardware acceleration support
- Quantization options
- Optimization for inference

```bash
# Basic ONNX deployment
modelgardener_cli.py deploy --format onnx

# Optimized ONNX with quantization
modelgardener_cli.py deploy \
    --format onnx \
    --optimize \
    --quantize int8 \
    --target-platform gpu
```

**ONNX Output:**
- `.onnx` model file
- Metadata and documentation
- Optimization reports
- Compatibility information

### TensorFlow Lite Deployment

**Features:**
- Mobile and edge optimization
- Ultra-low latency inference
- Memory efficiency
- Hardware delegation support

```bash
# TensorFlow Lite deployment
modelgardener_cli.py deploy --format tflite

# Mobile-optimized TFLite
modelgardener_cli.py deploy \
    --format tflite \
    --target-platform mobile \
    --quantize int8 \
    --optimize
```

**TFLite Output:**
- `.tflite` model file
- Representative dataset
- Quantization statistics
- Performance benchmarks

### TensorFlow.js Deployment

**Features:**
- Browser and Node.js compatibility
- Client-side inference
- WebGL acceleration
- Progressive loading

```bash
# TensorFlow.js deployment
modelgardener_cli.py deploy --format tfjs

# Web-optimized deployment
modelgardener_cli.py deploy \
    --format tfjs \
    --optimize \
    --compress \
    --quantize fp16
```

**TF.js Output:**
- Model JSON and weights
- JavaScript integration code
- Web interface templates
- Performance optimization guides

## Configuration File Structure

Deployment-specific configuration:

```yaml
deployment:
  model:
    model_path: "./logs/models/best_model.keras"
    weights_path: null
  
  formats:
    onnx:
      enable: true
      optimization:
        graph_optimization: true
        constant_folding: true
        operator_fusion: true
      quantization:
        type: "int8"
        calibration_dataset_size: 100
        representative_data: "./data/representative"
    
    tflite:
      enable: true
      optimization:
        optimize_for_size: true
        optimize_for_latency: true
        enable_gpu_delegate: false
      quantization:
        type: "int8"
        inference_type: "QUANTIZED_UINT8"
        representative_dataset: "./data/calibration"
    
    tfjs:
      enable: true
      optimization:
        weight_quantization: true
        activation_quantization: false
        skip_op_check: false
      quantization:
        type: "float16"
        dtype: "float16"
  
  security:
    encryption:
      enable: false
      algorithm: "AES-256"
      key_file: null
      auto_generate_key: true
    
    api_security:
      enable_api_key: false
      api_key_file: null
      rate_limiting: true
      cors_enabled: true
  
  serving:
    api_server:
      enable: false
      host: "0.0.0.0"
      port: 8000
      workers: 4
      timeout: 30
      max_requests_per_minute: 1000
    
    endpoints:
      prediction: "/predict"
      health: "/health"
      metrics: "/metrics"
      documentation: "/docs"
  
  optimization:
    target_platform: "cpu"
    batch_size: 1
    input_shape: null
    precision: "fp32"
    memory_optimization: true
    compute_optimization: true
  
  output:
    output_dir: "./deployment"
    package_deployment: false
    generate_docker: false
    include_documentation: true
    
    monitoring:
      enable_metrics: true
      performance_logging: true
      error_tracking: true
      usage_analytics: false
```

## Model Optimization

### Quantization Options

**INT8 Quantization:**
- 4x model size reduction
- Significant speedup on CPU
- Minimal accuracy loss
- Calibration dataset required

```bash
# INT8 quantization
modelgardener_cli.py deploy \
    --format onnx,tflite \
    --quantize int8 \
    --target-platform cpu
```

**FP16 Quantization:**
- 2x model size reduction
- GPU acceleration benefits
- Faster than INT8 on modern GPUs
- Better accuracy preservation

```bash
# FP16 quantization
modelgardener_cli.py deploy \
    --format onnx \
    --quantize fp16 \
    --target-platform gpu
```

**Dynamic Quantization:**
- Runtime quantization
- No calibration dataset needed
- Good balance of speed and accuracy
- Compatible with most platforms

### Graph Optimization

**Optimization Techniques:**
- Constant folding
- Dead code elimination
- Operator fusion
- Layout optimization
- Memory planning

**Platform-Specific Optimization:**
- CPU: SIMD instructions, threading
- GPU: Kernel fusion, memory coalescing
- Edge: Memory efficiency, power optimization
- Mobile: Latency optimization, battery efficiency

## Security Features

### Model Encryption

**Encryption Methods:**
- AES-256 encryption
- RSA key exchange
- Hardware security modules
- Key derivation functions

```bash
# Encrypt model with custom key
modelgardener_cli.py deploy \
    --format onnx \
    --encrypt \
    --encryption-key ./secure_keys/model.key
```

**Key Management:**
- Automatic key generation
- Secure key storage
- Key rotation support
- Hardware security integration

### API Security

**Security Features:**
- API key authentication
- Rate limiting
- CORS protection
- HTTPS encryption
- Request validation

```bash
# Secure API deployment
modelgardener_cli.py deploy \
    --serve \
    --secure-serving \
    --api-key ./keys/production.key
```

## REST API Server

### API Endpoints

**Prediction Endpoint:**
```
POST /predict
Content-Type: multipart/form-data or application/json

# File upload
curl -X POST \
  -F "image=@image.jpg" \
  http://localhost:8000/predict

# JSON input
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/image.jpg"}' \
  http://localhost:8000/predict
```

**Health Check:**
```
GET /health

Response:
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0",
  "uptime": 3600
}
```

**Metrics Endpoint:**
```
GET /metrics

Response:
{
  "total_requests": 1000,
  "average_response_time": 0.05,
  "error_rate": 0.01,
  "model_accuracy": 0.95
}
```

### API Documentation

Auto-generated API documentation includes:
- Interactive Swagger UI
- Request/response schemas
- Example requests
- Error codes and handling
- Performance guidelines

## Docker Integration

### Container Generation

```bash
# Generate Docker configuration
modelgardener_cli.py deploy \
    --format onnx \
    --docker \
    --serve \
    --package
```

**Generated Files:**
- `Dockerfile` - Container definition
- `docker-compose.yml` - Multi-service setup
- `requirements.txt` - Python dependencies
- `startup.sh` - Container startup script
- `.dockerignore` - Build optimization

### Container Features

**Base Images:**
- TensorFlow Serving images
- ONNX Runtime images
- Custom optimized images
- Security-hardened images

**Optimization:**
- Multi-stage builds
- Layer caching
- Minimal base images
- Security scanning

## Performance Monitoring

### Metrics Collection

**Performance Metrics:**
- Inference latency
- Throughput (requests/second)
- Memory usage
- CPU/GPU utilization
- Model accuracy

**System Metrics:**
- Request count
- Error rates
- Response times
- Resource utilization
- Network traffic

### Benchmarking

```bash
# Performance benchmarking
modelgardener_cli.py deploy \
    --format onnx \
    --target-platform cpu \
    --benchmark
```

**Benchmark Results:**
- Latency percentiles
- Throughput analysis
- Memory efficiency
- Platform comparison
- Optimization recommendations

## Output Structure

Deployment generates comprehensive outputs:

```
deployment/
├── models/
│   ├── model.onnx              # ONNX model
│   ├── model.tflite            # TensorFlow Lite model
│   ├── tfjs_model/             # TensorFlow.js model
│   │   ├── model.json
│   │   └── weights.bin
│   └── encrypted/              # Encrypted models
│       ├── model_encrypted.onnx
│       └── encryption_key.key
├── serving/
│   ├── api_server.py           # REST API server
│   ├── client_examples/        # Client code examples
│   ├── swagger.json            # API documentation
│   └── postman_collection.json # Postman collection
├── docker/
│   ├── Dockerfile              # Container definition
│   ├── docker-compose.yml      # Multi-service setup
│   ├── startup.sh              # Startup script
│   └── health_check.py         # Health check script
├── optimization/
│   ├── optimization_report.json # Optimization results
│   ├── quantization_analysis.json
│   ├── performance_benchmark.json
│   └── memory_analysis.json
├── security/
│   ├── encryption_keys/        # Generated keys
│   ├── api_keys/              # API authentication
│   ├── ssl_certificates/      # HTTPS certificates
│   └── security_audit.json    # Security analysis
├── documentation/
│   ├── deployment_guide.md     # Deployment instructions
│   ├── api_reference.md        # API documentation
│   ├── performance_guide.md    # Performance optimization
│   └── troubleshooting.md      # Common issues
└── packages/
    ├── deployment_package.tar.gz # Complete deployment package
    ├── model_only.zip          # Model files only
    └── api_server.zip          # Server components
```

## Platform-Specific Deployments

### Cloud Platforms

**AWS Deployment:**
- SageMaker integration
- Lambda functions
- ECS/EKS containers
- S3 model storage

**Google Cloud:**
- AI Platform integration
- Cloud Run deployment
- GKE containers
- Cloud Storage

**Azure:**
- Machine Learning Service
- Container Instances
- Kubernetes Service
- Blob Storage

### Edge Platforms

**Raspberry Pi:**
- ARM optimization
- TensorFlow Lite
- Minimal dependencies
- Power efficiency

**NVIDIA Jetson:**
- GPU acceleration
- TensorRT optimization
- CUDA integration
- Real-time inference

**Intel Neural Compute Stick:**
- OpenVINO integration
- USB deployment
- Edge inference
- Low power consumption

## Integration with Other Commands

### Complete Workflow

```bash
# Full ML pipeline
modelgardener_cli.py create ml_project
modelgardener_cli.py train --config config.yaml
modelgardener_cli.py evaluate --config config.yaml
modelgardener_cli.py deploy \
    --config config.yaml \
    --format all \
    --serve \
    --docker
```

### CI/CD Integration

```bash
# Automated deployment pipeline
modelgardener_cli.py deploy \
    --format onnx \
    --optimize \
    --quantize int8 \
    --package \
    --docker
```

## Best Practices

### Model Optimization

1. **Choose Appropriate Format:**
   - ONNX for cross-platform deployment
   - TFLite for mobile/edge devices
   - TF.js for web applications
   - Keras for TensorFlow ecosystem

2. **Quantization Strategy:**
   - Use representative calibration data
   - Validate accuracy after quantization
   - Consider platform-specific optimizations
   - Monitor performance trade-offs

3. **Security Considerations:**
   - Encrypt sensitive models
   - Use secure API authentication
   - Implement rate limiting
   - Monitor for anomalous usage

### Deployment Strategy

1. **Performance Testing:**
   - Benchmark on target hardware
   - Test with realistic workloads
   - Monitor resource usage
   - Validate accuracy preservation

2. **Monitoring and Maintenance:**
   - Implement comprehensive logging
   - Set up performance alerts
   - Plan for model updates
   - Monitor data drift

## Troubleshooting

### Common Issues

**Model Conversion Errors:**
```bash
# Check model compatibility
modelgardener_cli.py deploy --format onnx --validate

# Use specific ONNX opset
modelgardener_cli.py deploy --format onnx --opset-version 11
```

**Performance Issues:**
```bash
# Profile performance
modelgardener_cli.py deploy --format onnx --benchmark --profile

# Optimize for target platform
modelgardener_cli.py deploy \
    --format onnx \
    --target-platform cpu \
    --optimize
```

**Serving Errors:**
```bash
# Check server logs
modelgardener_cli.py deploy --serve --log-level DEBUG

# Test server endpoints
curl http://localhost:8000/health
```

### Debugging Options

```bash
# Verbose deployment
modelgardener_cli.py deploy --format onnx --verbose

# Validate deployment
modelgardener_cli.py deploy --format onnx --validate --benchmark
```

## See Also

- [Training Command](train.md)
- [Evaluation Command](evaluate.md)
- [Prediction Command](predict.md)
- [Docker Deployment Tutorial](../tutorials/docker-deployment.md)
- [Cloud Deployment Guide](../tutorials/cloud-deployment.md)
- [Model Optimization Guide](../tutorials/model-optimization.md)
