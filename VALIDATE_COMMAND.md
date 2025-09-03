# ModelGardener Configuration Checking

The ModelGardener CLI now includes a dedicated `check` command to check if your configuration files (JSON or YAML) are valid.

## Usage

### Basic Checking
```bash
python modelgardener_cli.py check config.yaml
python modelgardener_cli.py check config.json
```

### Verbose Checking
```bash
python modelgardener_cli.py check config.yaml --verbose
```

The verbose flag provides additional information including:
- Configuration file format (JSON/YAML)
- Configuration size
- Detailed configuration summary when checking passes

## Exit Codes
- `0`: Configuration is valid
- `1`: Configuration is invalid or file not found

## Examples

### Valid Configuration
```bash
$ python modelgardener_cli.py check cifar10_config.json
🔍 Checking configuration file: cifar10_config.json
✅ Configuration loaded from: cifar10_config.json
✅ Configuration validation passed
✅ Configuration file is valid!
```

### Invalid Configuration
```bash
$ python modelgardener_cli.py check invalid_config.yaml
🔍 Checking configuration file: invalid_config.yaml
✅ Configuration loaded from: invalid_config.yaml
❌ Missing required section: configuration
❌ Configuration file validation failed
```

### File Not Found
```bash
$ python modelgardener_cli.py check missing.yaml
🔍 Checking configuration file: missing.yaml
❌ Configuration file not found: missing.yaml
```

## Checking Process

The checking performs the following checks:
1. **File Loading**: Ensures the file exists and can be parsed as JSON/YAML
2. **Required Sections**: Checks for presence of required top-level sections:
   - `configuration`
   - `metadata`
3. **Configuration Structure**: Validates required subsections:
   - `task_type`
   - `data`
   - `model`
   - `training`
   - `runtime`
4. **Data Paths**: Warns if specified data directories don't exist

## Backward Compatibility

The original validation method still works:
```bash
python modelgardener_cli.py config --validate --config config.yaml
```

## Integration with Other Commands

The checking is automatically performed when:
- Training a model: `python modelgardener_cli.py train --config config.yaml`
- Evaluating a model: `python modelgardener_cli.py evaluate --config config.yaml`
