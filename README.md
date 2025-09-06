# ModelGardener

A command-line interface for deep learning model training with TensorFlow/Keras.

## Features

- Modular architecture for flexible model training
- Support for custom data loaders, models, and training loops
- Cross-validation and custom training strategies
- Runtime optimization with GPU and mixed precision support
- Configuration-driven training with YAML/JSON support

## Installation

```bash
pip install modelgardener
```

## Usage

```bash
# Run the CLI
mgd

# Or use the module directly
python -m modelgardener
```

## Development

This project uses `uv` for dependency management and building.

```bash
# Install dependencies
uv sync

# Build the package
uv build

# Run tests
uv run pytest
```

## License

MIT License
