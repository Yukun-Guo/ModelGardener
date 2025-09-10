# `deploy` Command

Deploy trained models to multiple formats with intelligent auto-discovery, optimization features, and enhanced parameter support.

## Synopsis

```bash
mg deploy [OPTIONS]
```

## Description

The `deploy` command provides comprehensive model deployment capabilities with enhanced auto-discovery:

- **Auto-Discovery**: Automatically finds config files and latest trained models
- **Multi-format Conversion**: Support for ONNX, TensorFlow Lite, TensorFlow.js, and Keras formats
- **Model Optimization**: Quantization support for edge deployment optimization
- **Security Features**: Model encryption and key management
- **Organized Output**: Structured deployment directories with clear naming
- **Short Parameters**: Intuitive short parameter support for faster workflows

## Auto-Discovery Features

### 🔍 Intelligent File Discovery
- **Config Discovery**: Automatically locates `config.yaml` in current directory
- **Model Discovery**: Finds the latest versioned model in `logs/` directory
  - Priority order: `final_model.keras` > `model.keras` > `best_model.keras` > latest timestamped model
- **Smart Defaults**: Uses ONNX and TFLite formats when none specified

### 📁 Enhanced Output Management
- **Organized Structure**: Creates deployment directories with clear format-based organization
- **Timestamped Results**: Optional timestamping for deployment versioning
- **Custom Directories**: Flexible output directory specification with `-o` option

## Options

### Core Options (All Optional with Auto-Discovery)

| Option | Short | Type | Description | Default |
|--------|-------|------|-------------|---------|
| `--config` | `-c` | `str` | Configuration file path | Auto-discovered `config.yaml` |
| `--model-path` | `-m` | `str` | Path to trained model file | Auto-discovered latest model |
| `--output-dir` | `-o` | `str` | Output directory for deployed models | `deployed_models` |

### Format and Optimization Options

| Option | Short | Type | Description | Default |
|--------|-------|------|-------------|---------|
| `--formats` | `-f` | `list` | Output formats (onnx, tflite, tfjs, keras) | `[onnx, tflite]` |
| `--quantize` | `-q` | `flag` | Apply quantization for ONNX/TFLite | False |

### Security Options

| Option | Short | Type | Description | Default |
|--------|-------|------|-------------|---------|
| `--encrypt` | `-e` | `flag` | Enable model encryption | False |
| `--encryption-key` | `-k` | `str` | Encryption key for model files | Auto-generate |

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

### Auto-Discovery Deployment (Recommended)

```bash
# Full auto-discovery with default formats (onnx, tflite)
mg deploy

# Auto-discovery with specific formats using short parameters
mg deploy -f keras tflite

# Auto-discovery with quantization
mg deploy -f tflite -q

# Auto-discovery with custom output directory
mg deploy -f onnx tflite -o production_models
```

### Format-Specific Deployment

```bash
# Single format deployment
mg deploy -f keras
mg deploy -f onnx -o onnx_models
mg deploy -f tflite -q -o mobile_models

# Multiple formats with optimization
mg deploy -f keras onnx tflite -q -o optimized_models

# Web deployment
mg deploy -f tfjs -o web_deployment
```

### Security and Optimization

```bash
# Encrypted deployment with auto-discovery
mg deploy -f onnx -e -k myencryptionkey

# Quantized deployment for edge devices
mg deploy -f tflite -q -o edge_deployment

# Secure multi-format deployment
mg deploy -f keras onnx -e -q -o secure_models
```

### Explicit Configuration

```bash
# Fully explicit deployment (overrides auto-discovery)
mg deploy -c config.yaml -m ./models/my_model.keras -f onnx -o custom_output

# Explicit with optimization
mg deploy -c ./configs/prod.yaml -m ./models/final.keras -f tflite -q

# Custom model with multiple formats
mg deploy -m ./models/custom_model.keras -f keras onnx tflite -o multi_format
```

### Production Deployment Workflows

```bash
# Mobile app deployment pipeline
mg deploy -f tflite -q -o mobile_app/models

# Web deployment pipeline
mg deploy -f tfjs -o web_app/public/models

# Multi-platform deployment
mg deploy -f keras onnx tflite tfjs -o cross_platform_models

# Versioned deployment
mg deploy -f onnx tflite -o models/v1.0 -q -e
```

### Complete Development Workflow

```bash
# Train, evaluate, and deploy pipeline
mg train
mg evaluate
mg deploy

# Custom pipeline with specific formats
mg train -c config.yaml
mg evaluate -c config.yaml
mg deploy -c config.yaml -f onnx tflite -q -o production

# Multi-model deployment comparison
mg deploy -m ./models/model_v1.keras -f onnx -o v1_deployment
mg deploy -m ./models/model_v2.keras -f onnx -o v2_deployment
```
    --host 0.0.0.0 \
    --port 443 \
    --workers 8 \
    --api-key production.key
```

### Containerized Deployment

```bash
# Docker deployment
mg deploy \
    --format onnx \
    --docker \
    --package \
    --serve

# Cloud-ready deployment
mg deploy \
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
mg deploy --format onnx

# Optimized ONNX with quantization
mg deploy \
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
mg deploy --format tflite

# Mobile-optimized TFLite
mg deploy \
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
mg deploy --format tfjs

# Web-optimized deployment
mg deploy \
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
mg deploy \
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
mg deploy \
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
mg deploy \
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
mg deploy \
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
mg deploy \
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
mg deploy \
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
mg create ml_project
mg train --config config.yaml
mg evaluate --config config.yaml
mg deploy \
    --config config.yaml \
    --format all \
    --serve \
    --docker
```

### CI/CD Integration

```bash
# Automated deployment pipeline
mg deploy \
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
mg deploy --format onnx --validate

# Use specific ONNX opset
mg deploy --format onnx --opset-version 11
```

**Performance Issues:**
```bash
# Profile performance
mg deploy --format onnx --benchmark --profile

# Optimize for target platform
mg deploy \
    --format onnx \
    --target-platform cpu \
    --optimize
```

**Serving Errors:**
```bash
# Check server logs
mg deploy --serve --log-level DEBUG

# Test server endpoints
curl http://localhost:8000/health
```

### Debugging Options

```bash
# Verbose deployment
mg deploy --format onnx --verbose

# Validate deployment
mg deploy --format onnx --validate --benchmark
```

## See Also

- [Training Command](train.md)
- [Evaluation Command](evaluate.md)
- [Prediction Command](predict.md)
- [Docker Deployment Tutorial](../tutorials/docker-deployment.md)
- [Cloud Deployment Guide](../tutorials/cloud-deployment.md)
- [Model Optimization Guide](../tutorials/model-optimization.md)
