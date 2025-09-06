# `check` Command

Perform comprehensive system checks, environment validation, and health monitoring for ModelGardener installations and projects.

## Synopsis

```bash
mg check [OPTIONS]
```

## Description

The `check` command provides thorough system and environment validation including:

- System requirements and dependencies verification
- Hardware capability assessment (CPU, GPU, Memory)
- Software environment validation (Python, TensorFlow, etc.)
- Project configuration and file integrity checks
- Performance benchmarking and optimization suggestions
- Security and permissions validation
- Integration testing and compatibility checks

## Options

### Check Scope

| Option | Short | Type | Description | Default |
|--------|-------|------|-------------|---------|
| `--all` | `-a` | `flag` | Run all available checks | False |
| `--system` | `-s` | `flag` | Check system requirements and hardware | False |
| `--environment` | `-e` | `flag` | Check software environment and dependencies | False |
| `--project` | `-p` | `str` | Check specific project directory | None |
| `--config` | `-c` | `str` | Check specific configuration file | None |

### Specific Checks

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--hardware` | `flag` | Check hardware capabilities | False |
| `--dependencies` | `flag` | Check software dependencies | False |
| `--permissions` | `flag` | Check file and directory permissions | False |
| `--performance` | `flag` | Run performance benchmarks | False |

### Output Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--output` | `str` | Output file for check results | None |
| `--format` | `str` | Output format (json, html, txt) | `txt` |
| `--verbose` | `flag` | Enable verbose output | False |
| `--fix` | `flag` | Attempt to fix detected issues | False |

### Filtering Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--level` | `str` | Check level (basic, standard, comprehensive) | `standard` |
| `--critical-only` | `flag` | Show only critical issues | False |
| `--warnings` | `flag` | Include warnings in output | True |
| `--suggestions` | `flag` | Include optimization suggestions | True |

## Usage Examples

### Basic System Checks

```bash
# Run all standard checks
mg check --all

# Check system and hardware only
mg check --system --hardware

# Check software environment
mg check --environment --dependencies
```

### Project-Specific Checks

```bash
# Check specific project
mg check --project ./my_ml_project/

# Check configuration file
mg check --config ./config.yaml

# Comprehensive project validation
mg check \
    --project ./my_project/ \
    --level comprehensive \
    --performance
```

### Targeted Checks

```bash
# Hardware capabilities only
mg check --hardware --performance

# Dependencies and permissions
mg check --dependencies --permissions

# Critical issues only
mg check --all --critical-only
```

### Output and Reporting

```bash
# Generate detailed HTML report
mg check \
    --all \
    --format html \
    --output health_report.html \
    --verbose

# JSON output for automation
mg check \
    --all \
    --format json \
    --output check_results.json
```

### Fix Issues Automatically

```bash
# Attempt to fix detected issues
mg check --all --fix

# Fix specific category
mg check --permissions --fix
```

## Check Categories

### System Requirements

**Hardware Checks:**
- CPU specifications and capabilities
- Memory (RAM) availability and usage
- Storage space and performance
- GPU detection and CUDA compatibility
- Network connectivity and bandwidth

**Operating System:**
- OS version and compatibility
- System libraries and drivers
- Environment variables
- System permissions and access rights

```bash
# Comprehensive system check
mg check --system --verbose
```

**Example Output:**
```
System Requirements Check:
==========================

✓ Operating System: Ubuntu 20.04.3 LTS (Compatible)
✓ CPU: Intel Core i7-10700K (8 cores, 16 threads)
✓ RAM: 32.0 GB available (16.0 GB recommended minimum)
✓ Storage: 500 GB available on primary drive
✓ Python: 3.9.7 (Compatible with TensorFlow 2.8+)

GPU Check:
✓ NVIDIA GeForce RTX 3080 detected
✓ CUDA Version: 11.4 (Compatible)
✓ cuDNN Version: 8.2.1 (Compatible)
✓ GPU Memory: 10 GB VRAM available

Network:
✓ Internet connectivity: Available
✓ Download speed: 100 Mbps
⚠ Upload speed: 10 Mbps (may be slow for cloud deployments)
```

### Software Environment

**Python Environment:**
- Python version compatibility
- Virtual environment detection
- Package manager availability
- Module import testing

**Dependencies:**
- TensorFlow/Keras installation
- Required packages and versions
- Optional packages for enhanced features
- Custom package compatibility

```bash
# Environment validation
mg check --environment --dependencies
```

**Example Output:**
```
Software Environment Check:
===========================

Python Environment:
✓ Python 3.9.7 (Recommended: 3.8+)
✓ Virtual environment: venv activated
✓ Package manager: pip 21.2.4

Core Dependencies:
✓ TensorFlow: 2.8.0 (Latest compatible)
✓ NumPy: 1.21.2 (Compatible)
✓ Pillow: 8.3.2 (Compatible)
✓ PyYAML: 5.4.1 (Compatible)
✓ Matplotlib: 3.4.3 (Compatible)

Optional Dependencies:
✓ ONNX: 1.10.2 (For model conversion)
✓ ONNXRuntime: 1.9.0 (For ONNX inference)
⚠ TensorFlow Lite: Not installed (for mobile deployment)
✗ Weights & Biases: Not installed (for experiment tracking)

Custom Modules:
✓ ModelGardener core modules importable
✓ Custom function loader working
✓ Configuration parser functional
```

### Project Validation

**Project Structure:**
- Directory organization and completeness
- Required files presence
- Configuration file validity
- Data directory structure
- Model and log directories

**Configuration Validation:**
- YAML syntax and structure
- Parameter value ranges
- Path existence and accessibility
- Resource requirements assessment

```bash
# Project structure validation
mg check --project ./ml_project/ --verbose
```

**Example Output:**
```
Project Validation Check:
=========================

Project Structure:
✓ Root directory: ./ml_project/
✓ Configuration file: config.yaml present
✓ Data directories: train/, val/, test/ present
✓ Custom modules: custom_modules/ directory found
✓ Requirements file: requirements.txt present
✓ Documentation: README.md present

Configuration Validation:
✓ YAML syntax: Valid
✓ Required sections: All present
✓ Data paths: All directories exist
✓ Model parameters: Within valid ranges
⚠ Batch size (128) may be too large for available memory
✓ Custom functions: All referenced functions found

Data Validation:
✓ Training data: 5,000 images found
✓ Validation data: 1,000 images found
✓ Test data: 1,000 images found
✓ Class balance: Acceptable (max deviation: 15%)
⚠ Image sizes vary (224x224 to 512x512)
✓ File formats: All JPEG/PNG (supported)
```

### Performance Benchmarking

**Hardware Performance:**
- CPU benchmark tests
- Memory throughput testing
- Storage I/O performance
- GPU compute capabilities
- Network bandwidth testing

**ML Performance:**
- TensorFlow operations benchmarking
- Matrix multiplication performance
- Data loading efficiency
- Model inference speed testing

```bash
# Performance benchmarking
mg check --performance --verbose
```

**Example Output:**
```
Performance Benchmark:
======================

CPU Performance:
✓ Single-core score: 1,250 (Excellent)
✓ Multi-core score: 8,500 (Excellent)
✓ Matrix multiplication: 15.2 GFLOPS
✓ Memory bandwidth: 45.6 GB/s

GPU Performance:
✓ CUDA cores: 8,704
✓ Tensor cores: 272 (2nd gen)
✓ Memory bandwidth: 760 GB/s
✓ FP32 performance: 29.8 TFLOPS
✓ Mixed precision: 119 TFLOPS

Storage Performance:
✓ Sequential read: 3,500 MB/s (NVMe SSD)
✓ Sequential write: 3,200 MB/s
✓ Random 4K read: 650,000 IOPS
✓ Random 4K write: 580,000 IOPS

ML Framework Performance:
✓ TensorFlow GPU acceleration: Working
✓ Sample model training: 2.3x speedup over CPU
✓ Data pipeline efficiency: 95% GPU utilization
✓ Mixed precision: 1.8x additional speedup
```

## Comprehensive Health Report

### Check Levels

**Basic Level:**
- Essential system requirements
- Core dependencies
- Basic configuration validation
- Critical error detection

**Standard Level (Default):**
- All basic checks
- Hardware capability assessment
- Extended dependency validation
- Project structure validation
- Performance overview

**Comprehensive Level:**
- All standard checks
- Detailed performance benchmarking
- Security and permissions audit
- Optimization recommendations
- Integration testing

```bash
# Comprehensive health check
mg check \
    --all \
    --level comprehensive \
    --format html \
    --output comprehensive_report.html
```

### Health Score and Recommendations

```
ModelGardener Health Report:
============================

Overall Health Score: 87/100 (Good)

Category Breakdown:
├── System Requirements: 95/100 (Excellent)
├── Software Environment: 88/100 (Good)
├── Project Configuration: 82/100 (Good)
├── Performance: 85/100 (Good)
└── Security: 78/100 (Satisfactory)

Critical Issues: 0
Warnings: 3
Suggestions: 7

Top Recommendations:
1. Install TensorFlow Lite for mobile deployment support
2. Reduce batch size from 128 to 64 for better memory efficiency
3. Enable mixed precision training for 1.8x speed improvement
4. Standardize image sizes in dataset for optimal preprocessing
5. Install Weights & Biases for enhanced experiment tracking

Estimated Performance Impact:
- Training speed: Could improve by 35%
- Memory usage: Could reduce by 25%
- Model accuracy: No negative impact expected
```

## Issue Detection and Resolution

### Common Issues and Fixes

**GPU Not Detected:**
```bash
# Check GPU availability
mg check --hardware --gpu

# Suggested fixes:
# 1. Update NVIDIA drivers
# 2. Install CUDA toolkit
# 3. Verify TensorFlow-GPU installation
```

**Memory Issues:**
```bash
# Check memory requirements
mg check --project ./project/ --performance

# Automatic fixes:
mg check --project ./project/ --fix
# - Reduces batch size automatically
# - Enables memory growth
# - Suggests model optimizations
```

**Dependency Conflicts:**
```bash
# Check for dependency issues
mg check --dependencies --verbose

# Suggested resolution steps provided
# Automatic virtual environment setup offered
```

### Automated Fixes

**Available Auto-fixes:**
- Dependency installation
- Configuration parameter adjustment
- Directory creation
- Permission corrections
- Environment variable setup

```bash
# Apply automatic fixes
mg check --all --fix --verbose
```

**Fix Report:**
```
Automated Fix Report:
=====================

Applied Fixes:
✓ Created missing logs/ directory
✓ Adjusted batch_size from 128 to 64 in config.yaml
✓ Set appropriate permissions on data/ directory
✓ Created backup of original configuration

Manual Actions Required:
⚠ Install TensorFlow Lite: pip install tensorflow-lite
⚠ Update NVIDIA drivers to version 470+
⚠ Consider upgrading RAM for optimal performance

All automatic fixes have been applied safely.
Backup files created with .backup extension.
```

## Integration and Monitoring

### CI/CD Integration

```bash
# Lightweight check for CI/CD
mg check \
    --environment \
    --critical-only \
    --format json \
    --output ci_check.json

# Return codes:
# 0 = All checks passed
# 1 = Warnings present
# 2 = Critical issues found
```

### Monitoring and Alerts

```bash
# Scheduled health monitoring
mg check \
    --all \
    --format json \
    --output /var/log/modelgardener/health_$(date +%Y%m%d).json

# Integration with monitoring systems
# Parse JSON output for metrics collection
```

## Best Practices

### Regular Health Checks

1. **Daily Checks:**
   - Basic system and environment validation
   - Project configuration verification
   - Critical issue detection

2. **Weekly Checks:**
   - Comprehensive performance benchmarking
   - Dependency update verification
   - Security audit

3. **Before Major Operations:**
   - Full system validation before training
   - Environment check before deployment
   - Project validation before production

### Proactive Monitoring

1. **Automated Scheduling:**
   - Set up cron jobs for regular checks
   - Integrate with monitoring systems
   - Configure alerting for critical issues

2. **Performance Tracking:**
   - Monitor performance trends over time
   - Track system degradation
   - Optimize based on recommendations

## Troubleshooting

### Common Check Failures

**Permission Denied Errors:**
```bash
# Check and fix permissions
mg check --permissions --fix

# Manual permission fix
sudo chown -R $USER:$USER /path/to/project/
chmod -R 755 /path/to/project/
```

**GPU Detection Issues:**
```bash
# Detailed GPU diagnostics
mg check --hardware --gpu --verbose

# Check NVIDIA installation
nvidia-smi
nvcc --version
```

**Dependency Resolution:**
```bash
# Clean dependency check
mg check --dependencies --verbose

# Fresh environment setup
python -m venv fresh_env
source fresh_env/bin/activate
pip install -r requirements.txt
```

## See Also

- [Installation Guide](../tutorials/installation.md)
- [System Requirements](../tutorials/system-requirements.md)
- [Troubleshooting Guide](../tutorials/troubleshooting.md)
- [Performance Optimization](../tutorials/performance-optimization.md)
- [Configuration Guide](../tutorials/configuration.md)
