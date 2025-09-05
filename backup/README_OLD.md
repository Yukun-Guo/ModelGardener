# ModelGardener Generated Scripts

Generated on: 2025-09-02 15:03:27

## Overview

This directory contains automatically generated Python scripts for training, evaluating, predicting, and deploying your create_simple_cnn model.

## Files

- `train.py` - Training script
- `evaluation.py` - Model evaluation script
- `prediction.py` - Prediction script for new images
- `deploy.py` - REST API deployment script
- `requirements.txt` - Required Python packages
- `README.md` - This file

## Setup

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Ensure your data is organized as specified in the configuration:
```
./data/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   └── ...
└── ...

./data/
├── class1/
├── class2/
└── ...

./data/test/  (for evaluation)
├── class1/
├── class2/
└── ...
```

## Usage

### Training

Train your model:
```bash
python train.py
```

The script will:
- Load the configuration from `model_config.yaml`
- Create data generators with augmentation
- Build and compile the model
- Train for 100 epochs
- Save the best model to the specified directory

### Evaluation

Evaluate your trained model:
```bash
python evaluation.py
```

The script will:
- Load the best trained model
- Evaluate on test data
- Generate classification report
- Create and save confusion matrix
- Save evaluation results

### Prediction

Make predictions on new images:

Single image:
```bash
python prediction.py --input path/to/image.jpg
```

Batch prediction:
```bash
python prediction.py --input path/to/images/directory/
```

Advanced options:
```bash
python prediction.py --input image.jpg --output results.json --top-k 3
```

### Deployment

Deploy your model as a REST API:
```bash
python deploy.py
```

Options:
```bash
python deploy.py --host 0.0.0.0 --port 8080 --debug
```

The API will be available at `http://localhost:5000` with the following endpoints:
- `GET /health` - Health check
- `POST /predict` - Prediction endpoint
- `GET /classes` - Available classes

## Customization

All scripts are generated based on your YAML configuration and can be customized as needed. The scripts include:

- Automatic configuration loading
- Error handling and logging
- Support for custom functions (if defined)
- Flexible input/output handling

## Custom Functions

If you have custom functions defined in your configuration, make sure they are available in the `src` directory relative to these scripts.

## Notes

- Models are saved in the directory specified in your configuration
- All scripts use the same preprocessing and normalization as specified in your configuration
- Cross-validation training is supported when enabled in the configuration
- The deployment script provides a simple REST API suitable for testing and development

## Support

These scripts are generated automatically by ModelGardener. For issues or customization needs, refer to the ModelGardener documentation.
