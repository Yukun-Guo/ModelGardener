# ModelGardener Documentation

Welcome to the comprehensive documentation for ModelGardener - a powerful deep learning framework with CLI interface, custom function support, and automated script generation.

## 📚 Documentation Structure

### Getting Started
- [Installation Guide](installation.md)
- [Quick Start Tutorial](tutorials/quickstart.md)
- [Project Structure Overview](project-structure.md)

### CLI Reference
- [CLI Overview](cli/README.md)
- [Configuration Management](cli/config.md)
- [Project Creation](cli/create.md)
- [Model Training](cli/train.md)
- [Model Evaluation](cli/evaluate.md)
- [Prediction](cli/predict.md)
- [Model Deployment](cli/deploy.md)
- [Model Listing](cli/models.md)
- [Configuration Validation](cli/check.md)

### Tutorials
- [Complete Workflow Tutorial](tutorials/complete-workflow.md)
- [Custom Functions Guide](tutorials/custom-functions.md)
- [Advanced Configuration](tutorials/advanced-configuration.md)
- [Production Deployment](tutorials/production-deployment.md)
- [Multi-Format Model Conversion](tutorials/model-conversion.md)

### API Reference
- [Script Generator API](api/script-generator.md)
- [Enhanced Trainer API](api/enhanced-trainer.md)
- [Configuration Manager API](api/config-manager.md)
- [Custom Function Wrappers](api/function-wrappers.md)

### Advanced Topics
- [Custom Model Integration](advanced/custom-models.md)
- [Custom Loss Functions](advanced/custom-losses.md)
- [Performance Optimization](advanced/performance.md)
- [Security Features](advanced/security.md)

## 🚀 Quick Navigation

### Most Common Tasks

| Task | Command | Documentation |
|------|---------|---------------|
| Create new project | `modelgardener_cli.py create` | [create command](cli/create.md) |
| Train model | `modelgardener_cli.py train` | [train command](cli/train.md) |
| Evaluate model | `modelgardener_cli.py evaluate` | [evaluate command](cli/evaluate.md) |
| Make predictions | `modelgardener_cli.py predict` | [predict command](cli/predict.md) |
| Deploy model | `modelgardener_cli.py deploy` | [deploy command](cli/deploy.md) |

### Generated Scripts

| Script | Purpose | Documentation |
|--------|---------|---------------|
| `train.py` | Standalone training | [Training Tutorial](tutorials/generated-scripts.md#training) |
| `evaluation.py` | Model evaluation | [Evaluation Tutorial](tutorials/generated-scripts.md#evaluation) |
| `prediction.py` | Batch prediction | [Prediction Tutorial](tutorials/generated-scripts.md#prediction) |
| `deploy.py` | Model deployment | [Deployment Tutorial](tutorials/generated-scripts.md#deployment) |

## 🎯 Features Overview

### Core Features
- **CLI Interface**: Comprehensive command-line interface for all operations
- **Custom Functions**: Support for custom models, losses, metrics, and data loaders
- **Script Generation**: Automatic generation of production-ready Python scripts
- **Multiple Formats**: Export models to ONNX, TensorFlow Lite, TensorFlow.js
- **Security**: Model encryption and secure deployment options

### Advanced Features
- **Cross-validation**: Built-in k-fold cross-validation support
- **Performance Monitoring**: Comprehensive metrics and timing information
- **Visualization**: Automatic generation of plots and charts
- **Batch Processing**: Optimized batch prediction and evaluation
- **Production Ready**: Robust error handling and logging

## 🔧 Requirements

### System Requirements
- Python 3.8 or higher
- TensorFlow 2.x
- Keras 3.x
- CUDA support (optional, for GPU acceleration)

### Optional Dependencies
- `tf2onnx` - For ONNX model conversion
- `onnxruntime` - For ONNX model inference and quantization
- `tensorflowjs` - For TensorFlow.js conversion
- `cryptography` - For model encryption
- `matplotlib` / `seaborn` - For advanced visualizations

## 🤝 Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on contributing to ModelGardener.

## 📄 License

ModelGardener is released under the MIT License. See [LICENSE](../LICENSE) for details.

## 🆘 Support

- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/Yukun-Guo/ModelGardener/issues)
- **Discussions**: Community discussions on [GitHub Discussions](https://github.com/Yukun-Guo/ModelGardener/discussions)
- **Documentation**: This documentation is continuously updated and improved

---

*Last updated: September 4, 2025*
