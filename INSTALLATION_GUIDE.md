# ModelGardener Installation & Usage Guide

## Installation Methods

### Method 1: Package Installation (Recommended)

Install ModelGardener as a Python package to use it system-wide:

```bash
# Clone the repository
git clone https://github.com/Yukun-Guo/ModelGardener.git
cd ModelGardener

# Install the package
pip install -e .  # Development mode (recommended for contributors)
# or
pip install .     # Regular installation
```

After installation, you can use ModelGardener in three ways:

### Method 2: Direct Repository Usage

If you prefer not to install the package:

```bash
# Clone the repository
git clone https://github.com/Yukun-Guo/ModelGardener.git
cd ModelGardener

# Install dependencies
pip install -r requirements.txt
```

## Usage Options

### 1. Python Module (Recommended) - `python -m modelgardener`

After installing the package, use as a Python module:

```bash
# Show help
python -m modelgardener --help

# List available models
python -m modelgardener models

# Create a new project
python -m modelgardener create my_classifier

# Configure training interactively
python -m modelgardener config --interactive --output my_classifier/config.yaml

# Train a model
python -m modelgardener train --config my_classifier/config.yaml
```

### 2. Console Command - `modelgardener`

After installing the package, use the console command:

```bash
# Show help
modelgardener --help

# List available models  
modelgardener models

# Create a new project
modelgardener create my_classifier

# Configure training
modelgardener config --interactive --output my_classifier/config.yaml

# Train a model
modelgardener train --config my_classifier/config.yaml
```

### 3. Direct Script - `python main.py`

From the repository directory:

```bash
# Show help
python main.py --help

# List available models
python main.py models

# Create a new project
python main.py create my_classifier

# Configure training
python main.py config --interactive --output my_classifier/config.yaml

# Train a model
python main.py train --config my_classifier/config.yaml
```

## Quick Start Examples

### Example 1: Basic Image Classification Project

```bash
# Method 1: Using python -m (recommended)
python -m modelgardener create image_classifier
cd image_classifier
python -m modelgardener config --interactive
python train.py

# Method 2: Using console command  
modelgardener create image_classifier
cd image_classifier
modelgardener config --interactive
python train.py

# Method 3: Direct script (from ModelGardener directory)
python main.py create image_classifier
cd image_classifier
python ../main.py config --interactive
python train.py
```

### Example 2: Custom Configuration

```bash
# Create project with custom settings
python -m modelgardener config \
  --train-dir ./data/train \
  --val-dir ./data/val \
  --model-family resnet \
  --model-name ResNet-50 \
  --epochs 100 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --output custom_config.yaml

# Train with the configuration
python -m modelgardener train --config custom_config.yaml
```

## Comparison of Usage Methods

| Method | Command | When to Use |
|--------|---------|-------------|
| Python Module | `python -m modelgardener` | ✅ **Recommended** - Works anywhere after installation |
| Console Command | `modelgardener` | ✅ Shortest command - Works anywhere after installation |
| Direct Script | `python main.py` | 🔧 Development/testing - Must be in repository directory |

## Advantages of Each Method

### Python Module (`python -m modelgardener`)
- ✅ **Standard Python practice** - follows `python -m venv`, `python -m pip` pattern
- ✅ **Explicit** - clearly shows you're running a Python module
- ✅ **Portable** - works on any system with Python
- ✅ **Virtual environment friendly** - uses the correct Python interpreter

### Console Command (`modelgardener`)
- ✅ **Shortest syntax** - quickest to type
- ✅ **Shell completion** - may support tab completion
- ✅ **Familiar** - works like other CLI tools

### Direct Script (`python main.py`)
- ✅ **Development** - useful when working on the codebase
- ✅ **No installation** - works without package installation
- ❌ **Limited scope** - must be in repository directory

## Recommendation

**For most users**: Use `python -m modelgardener` after installing the package with `pip install -e .`

This approach provides the best balance of explicitness, portability, and Python ecosystem conventions.
