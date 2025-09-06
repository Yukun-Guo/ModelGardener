#!/usr/bin/env python3
"""
ModelGardener CLI Entry Point
Provides command-line access to ModelGardener functionality without the GUI.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import yaml
import json
import numpy as np
from PIL import Image

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from .cli_config import ModelConfigCLI, create_argument_parser
    from .config_manager import ConfigManager
    from .enhanced_trainer import EnhancedTrainer
except ImportError as e:
    print(f"❌ Error importing required modules: {e}")
    print("💡 Make sure all CLI dependencies are installed:")
    print("   uv add inquirer tensorflow")
    sys.exit(1)


class ModelGardenerCLI:
    """Main CLI interface for ModelGardener."""
    
    def __init__(self):
        self.config_cli = ModelConfigCLI()
        self.config_manager = ConfigManager()

    def find_config_file(self, directory: str = ".") -> Optional[str]:
        """Find existing config file in the specified directory."""
        config_patterns = ["config.yaml", "config.yml", "model_config.yaml", "model_config.yml", 
                          "config.json", "model_config.json"]
        
        for pattern in config_patterns:
            config_path = Path(directory) / pattern
            if config_path.exists():
                return str(config_path)
        
        return None

    def modify_existing_config(self, config_file: str, interactive: bool = False, **kwargs) -> bool:
        """Modify an existing configuration file."""
        if not os.path.exists(config_file):
            print(f"❌ Configuration file not found: {config_file}")
            return False
        
        print(f"🔧 Modifying existing configuration: {config_file}")
        
        # Load existing configuration
        config = self.config_cli.load_config(config_file)
        if not config:
            print("❌ Failed to load existing configuration")
            return False
        
        print("✅ Existing configuration loaded")
        self.config_cli.display_config_summary(config)
        
        if interactive:
            print("\n🔄 Interactive configuration modification")
            print("=" * 50)
            
            # Run interactive configuration with existing config as base
            modified_config = self.config_cli.interactive_configuration_with_existing(config)
        else:
            print("\n⚡ Batch configuration modification")
            
            # Apply batch modifications to existing config
            modified_config = self.config_cli.batch_configuration_with_existing(config, kwargs)
        
        # Validate and save the modified configuration
        if self.config_cli.validate_config(modified_config):
            print("\n📋 Modified Configuration Summary:")
            self.config_cli.display_config_summary(modified_config)
            
            # Determine output format from file extension
            format_type = 'yaml' if config_file.endswith(('.yaml', '.yml')) else 'json'
            
            if self.config_cli.save_config(modified_config, config_file, format_type):
                print(f"✅ Configuration updated successfully: {config_file}")
                return True
            else:
                print("❌ Failed to save configuration")
                return False
        else:
            print("❌ Modified configuration validation failed")
            return False

    def run_training(self, config_file: str, **kwargs):
        """Run training using CLI configuration."""
        print(f"🚀 Starting ModelGardener training from CLI")
        print(f"📄 Configuration: {config_file}")
        
        if not os.path.exists(config_file):
            print(f"❌ Configuration file not found: {config_file}")
            return False
        
        try:
            # Load configuration
            config = self.config_cli.load_config(config_file)
            if not config:
                print("❌ Failed to load configuration")
                return False
            
            # Validate configuration
            if not self.config_cli.validate_config(config):
                print("❌ Configuration validation failed")
                return False
            
            # Extract the main configuration
            main_config = config.get('configuration', {})
            
            print("✅ Configuration loaded and validated")
            self.config_cli.display_config_summary(config)
            
            # Extract custom functions and handle None string case
            custom_functions = config.get('metadata', {}).get('custom_functions', {})
            if custom_functions == "None" or custom_functions == "none":
                custom_functions = {}
            
            # Initialize trainer
            trainer = EnhancedTrainer(
                config=main_config,
                custom_functions=custom_functions
            )
            
            # Run training
            trainer.train()
            # print("\n🏃 Starting training...")
            # success = trainer.train()
            
            # if success:
            #     print("✅ Training completed successfully!")
            #     return True
            # else:
            #     print("❌ Training failed")
            #     return False
                
        except Exception as e:
            print(f"❌ Error during training: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def run_evaluation(self, config_file: str, model_path: str = None, data_path: str = None, 
                      output_format: str = "yaml", save_results: bool = True):
        """Run enhanced model evaluation using CLI."""
        print(f"📊 Starting ModelGardener evaluation from CLI")
        print(f"📄 Configuration: {config_file}")
        if model_path:
            print(f"🤖 Model path: {model_path}")
        if data_path:
            print(f"📁 Data path: {data_path}")
        
        if not os.path.exists(config_file):
            print(f"❌ Configuration file not found: {config_file}")
            return False
        
        try:
            # Load configuration
            config = self.config_cli.load_config(config_file)
            if not config:
                print("❌ Failed to load configuration")
                return False
            
            main_config = config.get('configuration', {})
            
            # Use provided model path or default from config
            if model_path:
                main_config['runtime']['model_dir'] = model_path
            
            # Use provided data path for evaluation
            if data_path:
                main_config['data']['test_dir'] = data_path
            
            print("✅ Configuration loaded for evaluation")
            
            # Use EnhancedTrainer for consistent behavior
            trainer = EnhancedTrainer(
                config=main_config,
                custom_functions=config.get('metadata', {}).get('custom_functions', {})
            )
            
            # Load model if not already loaded
            if not trainer.model:
                print("🔄 Loading model for evaluation...")
                trainer._load_saved_model()
            
            # Run evaluation
            print("\n📈 Starting evaluation...")
            results = trainer.evaluate()
            
            if results:
                print("✅ Evaluation completed successfully!")
                print("\n📊 Results:")
                for metric, value in results.items():
                    print(f"  {metric}: {value:.4f}")
                
                # Save results if requested
                if save_results:
                    self._save_evaluation_results(results, main_config.get('runtime', {}).get('model_dir', './logs'), output_format)
                
                return True
            else:
                print("❌ Evaluation failed")
                return False
                
        except Exception as e:
            print(f"❌ Error during evaluation: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def run_prediction(self, config_file: str, input_path: str, model_path: str = None, 
                      output_path: str = None, top_k: int = 5, batch_size: int = 32):
        """Run prediction using CLI."""
        print(f"🔮 Starting ModelGardener prediction from CLI")
        print(f"📄 Configuration: {config_file}")
        print(f"📁 Input: {input_path}")
        if model_path:
            print(f"🤖 Model path: {model_path}")
        
        if not os.path.exists(config_file):
            print(f"❌ Configuration file not found: {config_file}")
            return False
        
        if not os.path.exists(input_path):
            print(f"❌ Input path not found: {input_path}")
            return False
        
        try:
            # Load configuration
            config = self.config_cli.load_config(config_file)
            if not config:
                print("❌ Failed to load configuration")
                return False
            
            main_config = config.get('configuration', {})
            
            # Use provided model path or default from config
            if model_path:
                main_config['runtime']['model_dir'] = model_path
            
            print("✅ Configuration loaded for prediction")
            
            # Use EnhancedTrainer for consistent behavior
            trainer = EnhancedTrainer(
                config=main_config,
                custom_functions=config.get('metadata', {}).get('custom_functions', {})
            )
            
            # Load model if not already loaded
            if not trainer.model:
                print("🔄 Loading model for prediction...")
                trainer._load_saved_model()
            
            # Run prediction
            print(f"\n🔮 Starting prediction on {input_path}...")
            results = self._run_prediction_on_path(trainer, input_path, top_k, batch_size, main_config)
            
            if results:
                print("✅ Prediction completed successfully!")
                
                # Save results if output path specified
                if output_path:
                    self._save_prediction_results(results, output_path)
                    print(f"💾 Results saved to: {output_path}")
                
                return True
            else:
                print("❌ Prediction failed")
                return False
                
        except Exception as e:
            print(f"❌ Error during prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def run_deployment(self, config_file: str, model_path: str = None, output_formats: List[str] = None,
                      quantize: bool = False, encrypt: bool = False, encryption_key: str = None):
        """Run enhanced model deployment with multiple format support."""
        print(f"🚀 Starting ModelGardener deployment from CLI")
        print(f"📄 Configuration: {config_file}")
        if model_path:
            print(f"🤖 Model path: {model_path}")
        if output_formats:
            print(f"📦 Output formats: {', '.join(output_formats)}")
        
        if not os.path.exists(config_file):
            print(f"❌ Configuration file not found: {config_file}")
            return False
        
        try:
            # Load configuration
            config = self.config_cli.load_config(config_file)
            if not config:
                print("❌ Failed to load configuration")
                return False
            
            main_config = config.get('configuration', {})
            
            # Use provided model path or default from config
            if model_path:
                main_config['runtime']['model_dir'] = model_path
            
            print("✅ Configuration loaded for deployment")
            
            # Use EnhancedTrainer for consistent behavior
            trainer = EnhancedTrainer(
                config=main_config,
                custom_functions=config.get('metadata', {}).get('custom_functions', {})
            )
            
            # Load model if not already loaded
            if not trainer.model:
                print("🔄 Loading model for deployment...")
                trainer._load_saved_model()
            
            # Run deployment
            print("\n🚀 Starting model deployment...")
            success = self._deploy_model_formats(trainer, main_config, output_formats, quantize, encrypt, encryption_key)
            
            if success:
                print("✅ Deployment completed successfully!")
                return True
            else:
                print("❌ Deployment failed")
                return False
                
        except Exception as e:
            print(f"❌ Error during deployment: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def _save_evaluation_results(self, results: Dict[str, float], model_dir: str, output_format: str):
        """Save evaluation results to file."""
        try:
            os.makedirs(model_dir, exist_ok=True)
            
            if output_format.lower() == 'json':
                results_path = os.path.join(model_dir, 'evaluation_results.json')
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2)
            else:  # yaml
                results_path = os.path.join(model_dir, 'evaluation_results.yaml')
                with open(results_path, 'w') as f:
                    yaml.dump(results, f, default_flow_style=False)
            
            print(f"💾 Evaluation results saved to: {results_path}")
            
        except Exception as e:
            print(f"⚠️ Could not save evaluation results: {str(e)}")

    def _run_prediction_on_path(self, trainer, input_path: str, top_k: int, batch_size: int, config: Dict[str, Any]):
        """Run prediction on a file or directory."""
        try:
            # Get target size from config
            input_shape = config.get('model', {}).get('model_parameters', {}).get('input_shape', {})
            target_size = (input_shape.get('height', 224), input_shape.get('width', 224))
            
            # Get class labels if available
            class_labels = self._get_class_labels(config)
            
            if os.path.isfile(input_path):
                # Single file prediction
                return self._predict_single_file(trainer, input_path, target_size, class_labels, top_k)
            elif os.path.isdir(input_path):
                # Directory prediction
                return self._predict_directory(trainer, input_path, target_size, class_labels, top_k, batch_size)
            else:
                print(f"❌ Invalid input path: {input_path}")
                return None
                
        except Exception as e:
            print(f"❌ Error during prediction: {str(e)}")
            return None

    def _predict_single_file(self, trainer, image_path: str, target_size: tuple, class_labels: List[str], top_k: int):
        """Predict on a single image file."""
        try:
            # Load and preprocess image
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img = img.resize(target_size)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            predictions = trainer.model.predict(img_array, verbose=0)
            
            # Get top-k results
            top_indices = np.argsort(predictions[0])[::-1][:top_k]
            results = []
            
            for i, idx in enumerate(top_indices):
                class_name = class_labels[idx] if idx < len(class_labels) else f"class_{idx}"
                confidence = float(predictions[0][idx])
                results.append({
                    'rank': i + 1,
                    'class': class_name,
                    'confidence': confidence
                })
                print(f"  {i+1}. {class_name}: {confidence:.4f}")
            
            return {'image_path': image_path, 'predictions': results}
            
        except Exception as e:
            print(f"❌ Error predicting {image_path}: {str(e)}")
            return None

    def _predict_directory(self, trainer, directory: str, target_size: tuple, class_labels: List[str], top_k: int, batch_size: int):
        """Predict on all images in a directory."""
        try:
            # Find all image files
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            image_files = []
            
            for ext in extensions:
                image_files.extend(Path(directory).glob(f'*{ext}'))
                image_files.extend(Path(directory).glob(f'*{ext.upper()}'))
            
            if not image_files:
                print(f"❌ No images found in {directory}")
                return None
            
            print(f"📁 Found {len(image_files)} images")
            results = []
            
            # Process images
            for image_file in image_files:
                print(f"\n📷 Processing: {image_file.name}")
                result = self._predict_single_file(trainer, str(image_file), target_size, class_labels, top_k)
                if result:
                    results.append(result)
            
            return {'directory': directory, 'results': results, 'total_images': len(image_files)}
            
        except Exception as e:
            print(f"❌ Error predicting directory {directory}: {str(e)}")
            return None

    def _get_class_labels(self, config: Dict[str, Any]) -> List[str]:
        """Get class labels from config or generate default ones."""
        try:
            # Try to get from config
            num_classes = config.get('model', {}).get('model_parameters', {}).get('classes', 10)
            return [f"class_{i}" for i in range(num_classes)]
        except:
            return [f"class_{i}" for i in range(10)]  # Default

    def _save_prediction_results(self, results: Dict[str, Any], output_path: str):
        """Save prediction results to file."""
        try:
            if output_path.endswith('.json'):
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
            else:
                with open(output_path, 'w') as f:
                    yaml.dump(results, f, default_flow_style=False)
                    
        except Exception as e:
            print(f"⚠️ Could not save prediction results: {str(e)}")

    def _deploy_model_formats(self, trainer, config: Dict[str, Any], output_formats: List[str], 
                             quantize: bool, encrypt: bool, encryption_key: str) -> bool:
        """Deploy model in multiple formats."""
        try:
            model_dir = config.get('runtime', {}).get('model_dir', './logs')
            deploy_dir = os.path.join(model_dir, 'deployment')
            os.makedirs(deploy_dir, exist_ok=True)
            
            success = True
            
            # Default formats if none specified
            if not output_formats:
                output_formats = ['onnx', 'tflite']
            
            for format_name in output_formats:
                print(f"🔄 Converting to {format_name.upper()}...")
                
                if format_name.lower() == 'onnx':
                    success &= self._convert_to_onnx(trainer.model, deploy_dir, quantize)
                elif format_name.lower() == 'tflite':
                    success &= self._convert_to_tflite(trainer.model, deploy_dir, quantize)
                elif format_name.lower() == 'tfjs':
                    success &= self._convert_to_tfjs(trainer.model, deploy_dir)
                elif format_name.lower() == 'keras':
                    success &= self._save_keras_model(trainer.model, deploy_dir, encrypt, encryption_key)
                else:
                    print(f"⚠️ Unsupported format: {format_name}")
                    
            return success
            
        except Exception as e:
            print(f"❌ Error during deployment: {str(e)}")
            return False

    def _convert_to_onnx(self, model, deploy_dir: str, quantize: bool) -> bool:
        """Convert model to ONNX format."""
        try:
            import tf2onnx
            import onnx
            
            output_path = os.path.join(deploy_dir, 'model.onnx')
            
            # Convert to ONNX
            model_proto, _ = tf2onnx.convert.from_keras(model, output_path=output_path)
            
            if quantize:
                print("🔄 Quantizing ONNX model...")
                try:
                    from onnxruntime.quantization import quantize_dynamic, QuantType
                    quantized_path = os.path.join(deploy_dir, 'model_quantized.onnx')
                    quantize_dynamic(output_path, quantized_path, weight_type=QuantType.QUInt8)
                    print(f"✅ Quantized ONNX model saved: {quantized_path}")
                except ImportError:
                    print("⚠️ onnxruntime not available for quantization")
            
            print(f"✅ ONNX model saved: {output_path}")
            return True
            
        except ImportError:
            print("❌ tf2onnx not installed. Install with: pip install tf2onnx")
            return False
        except Exception as e:
            print(f"❌ Error converting to ONNX: {str(e)}")
            return False

    def _convert_to_tflite(self, model, deploy_dir: str, quantize: bool) -> bool:
        """Convert model to TensorFlow Lite format."""
        try:
            import tensorflow as tf
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            if quantize:
                print("🔄 Applying quantization...")
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
            
            tflite_model = converter.convert()
            
            # Save model
            suffix = '_quantized' if quantize else ''
            output_path = os.path.join(deploy_dir, f'model{suffix}.tflite')
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            print(f"✅ TFLite model saved: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error converting to TFLite: {str(e)}")
            return False

    def _convert_to_tfjs(self, model, deploy_dir: str) -> bool:
        """Convert model to TensorFlow.js format."""
        try:
            import tensorflowjs as tfjs
            
            output_path = os.path.join(deploy_dir, 'tfjs_model')
            tfjs.converters.save_keras_model(model, output_path)
            
            print(f"✅ TensorFlow.js model saved: {output_path}")
            return True
            
        except ImportError:
            print("❌ tensorflowjs not installed. Install with: pip install tensorflowjs")
            return False
        except Exception as e:
            print(f"❌ Error converting to TensorFlow.js: {str(e)}")
            return False

    def _save_keras_model(self, model, deploy_dir: str, encrypt: bool, encryption_key: str) -> bool:
        """Save Keras model with optional encryption."""
        try:
            output_path = os.path.join(deploy_dir, 'model.keras')
            model.save(output_path)
            
            if encrypt and encryption_key:
                print("🔄 Encrypting model...")
                encrypted_path = os.path.join(deploy_dir, 'model_encrypted.keras')
                self._encrypt_model_file(output_path, encrypted_path, encryption_key)
                print(f"✅ Encrypted model saved: {encrypted_path}")
            
            print(f"✅ Keras model saved: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving Keras model: {str(e)}")
            return False

    def _encrypt_model_file(self, input_path: str, output_path: str, key: str):
        """Encrypt model file using simple XOR encryption."""
        try:
            from cryptography.fernet import Fernet
            import base64
            
            # Generate key from provided string
            key_bytes = key.encode('utf-8')
            # Pad or truncate to 32 bytes for Fernet
            key_bytes = key_bytes[:32].ljust(32, b'\0')
            key_b64 = base64.urlsafe_b64encode(key_bytes)
            
            fernet = Fernet(key_b64)
            
            # Encrypt file
            with open(input_path, 'rb') as f:
                data = f.read()
            
            encrypted_data = fernet.encrypt(data)
            
            with open(output_path, 'wb') as f:
                f.write(encrypted_data)
                
            print("🔐 Model encrypted successfully")
            
        except ImportError:
            print("⚠️ cryptography not available, using simple XOR encryption")
            # Fallback to simple XOR
            with open(input_path, 'rb') as f:
                data = f.read()
            
            key_bytes = key.encode('utf-8')
            encrypted = bytearray()
            for i, byte in enumerate(data):
                encrypted.append(byte ^ key_bytes[i % len(key_bytes)])
            
            with open(output_path, 'wb') as f:
                f.write(encrypted)
                
        except Exception as e:
            print(f"⚠️ Error encrypting model: {str(e)}")

    def list_available_models(self):
        """List all available models."""
        print("🤖 Available Models in ModelGardener")
        print("=" * 50)
        
        for family, models in self.config_cli.available_models.items():
            print(f"\n📂 {family.upper()}")
            for model in models:
                print(f"  • {model}")

    def create_project_template(self, project_name: str = None, project_dir: str = ".", interactive: bool = False, **kwargs):
        """Create a new project template with CLI configuration."""
        
        # Create project path - if project_name is provided, create a subfolder
        if project_name:
            project_path = Path(project_dir) / project_name
            project_path.mkdir(parents=True, exist_ok=True)
        else:
            # If no project name provided, use current directory
            project_path = Path(project_dir).resolve()
            project_name = project_path.name
        
        print(f"🌱 Creating ModelGardener project: {project_name}")
        print(f"📁 Project directory: {project_path}")

        # Create project structure
        (project_path / "data" / "train").mkdir(parents=True, exist_ok=True)
        (project_path / "data" / "val").mkdir(parents=True, exist_ok=True)
        (project_path / "logs").mkdir(exist_ok=True)
        (project_path / "custom_modules").mkdir(exist_ok=True)

        # Create configuration based on mode
        config_file = project_path / "config.yaml"

        if interactive:
            print("\n🔧 Interactive project configuration")
            print("=" * 50)

            # Update paths to be relative to the project directory
            old_cwd = os.getcwd()
            os.chdir(project_path)

            try:
                config = self.config_cli.interactive_configuration()

                # Validate and save configuration using relative path
                config_file_relative = "config.yaml"
                if self.config_cli.validate_config(config):
                    self.config_cli.display_config_summary(config)
                    if self.config_cli.save_config(config, config_file_relative, 'yaml'):
                        print(f"✅ Configuration saved to {config_file}")
                else:
                    print("❌ Configuration validation failed, using default template")
                    self.config_cli.create_template(config_file_relative, 'yaml')
            finally:
                os.chdir(old_cwd)
        else:
            # Batch mode: create config using provided arguments or default template
            if kwargs:
                # Create a default config first, then apply batch modifications
                default_config = self.config_cli.create_default_config()
                config = self.config_cli.batch_configuration_with_existing(default_config, kwargs)
                if self.config_cli.validate_config(config):
                    self.config_cli.display_config_summary(config)
                    if self.config_cli.save_config(config, str(config_file), 'yaml'):
                        print(f"✅ Configuration saved to {config_file}")
                    else:
                        print("❌ Failed to save configuration, using default template")
                        self.config_cli.create_template(str(config_file), 'yaml')
                else:
                    print("❌ Configuration validation failed, using default template")
                    self.config_cli.create_template(str(config_file), 'yaml')
            else:
                # No batch args, create default template
                self.config_cli.create_template(str(config_file), 'yaml')
        
        # Create README
        readme_content = f"""# {project_name} - ModelGardener Project

## Project Structure
```
{project_name}/
├── data/
│   ├── train/          # Training data
│   └── val/            # Validation data
├── logs/               # Training logs and models
├── custom_modules/     # Custom functions (models, losses, etc.)
├── config.yaml         # Model configuration
├── train.py           # Training script (auto-generated)
├── evaluation.py      # Evaluation script (auto-generated)
├── prediction.py      # Prediction script (auto-generated)
├── deploy.py          # Deployment script (auto-generated)
├── requirements.txt   # Python dependencies (auto-generated)
└── README.md          # This file
```

## Quick Start

### 1. Prepare Your Data
Place your training images in `data/train/` and validation images in `data/val/`

### 2. Configure Your Model
Edit the `config.yaml` file to customize your model settings, or use the interactive configuration:
```bash
# Interactive configuration (overwrites config.yaml)
python /path/to/ModelGardener/modelgardener_cli.py config --interactive --output config.yaml

# Or directly edit config.yaml
```

### 3. Train Your Model
```bash
# Use the generated training script
python train.py

# Or use the CLI
python /path/to/ModelGardener/modelgardener_cli.py train --config config.yaml
```

### 4. Evaluate Your Model
```bash
# Use the generated evaluation script  
python evaluation.py

# Or use the CLI
python /path/to/ModelGardener/modelgardener_cli.py evaluate --config config.yaml --model-path logs/final_model.keras
```

## Generated Files

This project includes auto-generated files to help you get started:

- **config.yaml** - Complete model configuration with examples and documentation
- **train.py** - Ready-to-use training script
- **evaluation.py** - Model evaluation script
- **prediction.py** - Inference script for new data
- **deploy.py** - Deployment utilities

## Configuration Options

The `config.yaml` file includes comprehensive settings for:
- Model architecture selection (ResNet, EfficientNet, Custom, etc.)  
- Training parameters (epochs, learning rate, batch size, etc.)
- Data preprocessing and augmentation options
- Runtime settings (GPU usage, model directory, etc.)
- Custom function integration

## Custom Functions

You can customize any aspect of the training pipeline by creating your own Python files:
1. Create Python files with your custom functions (models, loss functions, etc.)
2. Update the `config.yaml` to reference your custom function files
3. The training scripts will automatically load and use your custom functions

## Need Help?

- Run ModelGardener CLI with `--help` to see all available options
- Use interactive mode for guided configuration: `modelgardener_cli.py config --interactive`
- Check the custom_modules/README.md for detailed examples
- See the ModelGardener documentation for advanced usage
"""
        
        readme_file = project_path / "README.md"
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        
        print(f"✅ Project template created successfully!")
        print(f"📖 See {readme_file} for instructions")

    def check_configuration(self, config_file: str, verbose: bool = False) -> bool:
        """Check and validate a configuration file."""
        print(f"🔍 Checking configuration file: {config_file}")
        
        if not os.path.exists(config_file):
            print(f"❌ Configuration file not found: {config_file}")
            return False
        
        try:
            # Load configuration
            config = self.config_cli.load_config(config_file)
            if not config:
                print("❌ Failed to load configuration file")
                return False
            
            if verbose:
                print(f"📄 Configuration file format: {'YAML' if config_file.endswith(('.yaml', '.yml')) else 'JSON'}")
                print(f"📊 Configuration size: {len(str(config))} characters")
            
            # Validate configuration
            is_valid = self.config_cli.validate_config(config)
            
            if verbose and is_valid:
                # Display configuration summary
                print("\n📋 Configuration Summary:")
                self.config_cli.display_config_summary(config)
            
            if is_valid:
                print("✅ Configuration file is valid!")
            else:
                print("❌ Configuration file validation failed")
            
            return is_valid
            
        except Exception as e:
            print(f"❌ Error validating configuration: {str(e)}")
            if verbose:
                import traceback
                traceback.print_exc()
            return False

    def preview_data(self, config_file: str, num_samples: int = 8, split: str = 'train', 
                     save_plot: bool = False, output_path: str = None):
        """Preview data samples with visualization including preprocessing and augmentation."""
        import matplotlib.pyplot as plt
        import numpy as np
        import tensorflow as tf
        from pathlib import Path
        
        print(f"🔍 Previewing data from configuration: {config_file}")
        
        if not os.path.exists(config_file):
            print(f"❌ Configuration file not found: {config_file}")
            return False
        
        try:
            # Load configuration
            config = self.config_cli.load_config(config_file)
            if not config:
                print("❌ Failed to load configuration file")
                return False
            
            main_config = config.get('configuration', {})
            data_config = main_config.get('data', {})
            
            # Get data loader configuration
            data_loader_config = data_config.get('data_loader', {})
            selected_loader = data_loader_config.get('selected_data_loader', 'Default')
            
            # Get preprocessing and augmentation configs
            preprocessing_config = data_config.get('preprocessing', {})
            augmentation_config = data_config.get('augmentation', {})
            
            print(f"📊 Data loader: {selected_loader}")
            print(f"🎯 Split: {split}")
            print(f"📈 Number of samples to preview: {num_samples}")
            print(f"🔧 Preprocessing enabled: {any(v.get('enabled', False) for v in preprocessing_config.values() if isinstance(v, dict))}")
            print(f"🎨 Augmentation enabled: {any(v.get('enabled', False) for v in augmentation_config.values() if isinstance(v, dict))}")
            
            # Load raw data first
            preview_images_raw = None
            preview_labels = None
            class_names = None
            
            if selected_loader == 'Custom_load_cifar10_npz_data':
                # Handle CIFAR-10 NPZ data loader
                npz_file_path = data_loader_config.get('parameters', {}).get('npz_file_path', './data/cifar10.npz')
                
                print(f"📂 Loading data from: {npz_file_path}")
                
                if not os.path.exists(npz_file_path):
                    print(f"❌ NPZ file not found: {npz_file_path}")
                    return False
                
                # Load NPZ data directly (raw, unnormalized)
                data = np.load(npz_file_path)
                x_data = data['x'].astype(np.float32)  # Keep raw 0-255 values initially
                y_data = data['y']
                
                print(f"📊 Loaded data shape: {x_data.shape}")
                print(f"🎯 Labels shape: {y_data.shape}")
                print(f"🏷️ Unique classes: {np.unique(y_data)}")
                
                # Get subset for preview
                indices = np.random.choice(len(x_data), min(num_samples, len(x_data)), replace=False)
                preview_images_raw = x_data[indices] / 255.0  # Normalize for display
                preview_labels = y_data[indices]
                
                # CIFAR-10 class names
                class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                              'dog', 'frog', 'horse', 'ship', 'truck']
                              
            elif selected_loader in ['ImageDataLoader', 'Default']:
                # Handle directory-based image data
                train_dir = data_config.get('train_dir', './data/train')
                val_dir = data_config.get('val_dir', './data/val')
                
                data_dir = train_dir if split == 'train' else val_dir
                print(f"📂 Loading images from directory: {data_dir}")
                
                if not os.path.exists(data_dir):
                    print(f"❌ Data directory not found: {data_dir}")
                    return False
                
                # Get class directories
                class_dirs = [d for d in os.listdir(data_dir) 
                             if os.path.isdir(os.path.join(data_dir, d))]
                class_dirs.sort()
                
                if not class_dirs:
                    print(f"⚠️ No class directories found in {data_dir}")
                    return False
                
                print(f"🏷️ Found classes: {class_dirs}")
                
                # Collect sample images
                sample_files = []
                sample_labels = []
                
                for class_idx, class_name in enumerate(class_dirs):
                    class_path = os.path.join(data_dir, class_name)
                    image_files = [f for f in os.listdir(class_path) 
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
                    
                    # Take a few samples from each class
                    samples_per_class = max(1, num_samples // len(class_dirs))
                    selected_files = np.random.choice(image_files, 
                                                    min(samples_per_class, len(image_files)), 
                                                    replace=False)
                    
                    for img_file in selected_files:
                        sample_files.append(os.path.join(class_path, img_file))
                        sample_labels.append(class_idx)
                
                # Shuffle and limit to num_samples
                combined = list(zip(sample_files, sample_labels))
                np.random.shuffle(combined)
                combined = combined[:num_samples]
                sample_files, sample_labels = zip(*combined) if combined else ([], [])
                
                # Load images
                preview_images_list = []
                preview_labels = list(sample_labels)
                
                for img_path in sample_files:
                    try:
                        from PIL import Image
                        img = Image.open(img_path)
                        img = img.convert('RGB')
                        img_array = np.array(img) / 255.0
                        preview_images_list.append(img_array)
                    except Exception as e:
                        print(f"⚠️ Failed to load image {img_path}: {e}")
                        continue
                
                preview_images_raw = np.array(preview_images_list) if preview_images_list else None
                class_names = class_dirs
                
            else:
                print(f"⚠️ Data loader '{selected_loader}' preview not yet implemented")
                print("💡 Currently supported: Custom_load_cifar10_npz_data, ImageDataLoader, Default")
                return False
            
            if preview_images_raw is None or len(preview_images_raw) == 0:
                print("❌ No images found to preview")
                return False
            
            # Apply preprocessing and augmentation to create processed images
            print(f"🔄 Applying preprocessing and augmentation...")
            preview_images_processed = self._apply_preprocessing_and_augmentation(
                preview_images_raw.copy(), preprocessing_config, augmentation_config, split == 'train'
            )
            
            # Create visualization showing both original and processed images
            print(f"🎨 Creating comparison visualization...")
            
            # Calculate grid size - we'll show original and processed side by side
            actual_samples = len(preview_images_raw)
            cols = min(4, actual_samples)  # Number of sample pairs per row
            rows = (actual_samples + cols - 1) // cols
            
            # Create subplots: 2 rows for each sample row (original + processed)
            fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 3, rows * 5))
            
            # Handle single sample case
            if rows == 1 and cols == 1:
                axes = np.array([[axes[0]], [axes[1]]])
            elif rows == 1:
                axes = axes.reshape(2, cols)
            elif cols == 1:
                axes = axes.reshape(rows * 2, 1)
            
            for i in range(actual_samples):
                row = i // cols
                col = i % cols
                
                # Original image (top row for this sample)
                orig_ax = axes[row * 2, col]
                if len(preview_images_raw[i].shape) == 3:  # RGB image
                    orig_ax.imshow(np.clip(preview_images_raw[i], 0, 1))
                else:  # Grayscale
                    orig_ax.imshow(np.clip(preview_images_raw[i], 0, 1), cmap='gray')
                
                # Add label information
                label_idx = preview_labels[i]
                if class_names and label_idx < len(class_names):
                    label_text = f"{class_names[label_idx]}"
                else:
                    label_text = f"Class {label_idx}"
                
                orig_ax.set_title(f"Original: {label_text}", fontsize=9)
                orig_ax.axis('off')
                
                # Processed image (bottom row for this sample)
                proc_ax = axes[row * 2 + 1, col]
                if len(preview_images_processed[i].shape) == 3:  # RGB image
                    proc_ax.imshow(np.clip(preview_images_processed[i], 0, 1))
                else:  # Grayscale
                    proc_ax.imshow(np.clip(preview_images_processed[i], 0, 1), cmap='gray')
                
                proc_ax.set_title(f"Processed: {label_text}", fontsize=9)
                proc_ax.axis('off')
                
                # Print label to terminal as well
                print(f"Sample {i+1}: {label_text}")
            
            # Hide empty subplots
            total_plots = rows * 2 * cols
            for i in range(actual_samples, cols * rows):
                row = i // cols
                col = i % cols
                axes[row * 2, col].axis('off')
                axes[row * 2 + 1, col].axis('off')
            
            plt.tight_layout()
            plt.suptitle(f'Data Preview: Original vs Processed - {split.capitalize()} Set ({selected_loader})', 
                        fontsize=12, y=0.98)
            
            # Save or show plot
            if save_plot:
                if output_path is None:
                    output_path = f"data_preview_{split}_{selected_loader}_comparison.png"
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"💾 Plot saved to: {output_path}")
            else:
                plt.show()
            
            print("✅ Data preview completed!")
            return True
            
        except Exception as e:
            print(f"❌ Error previewing data: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def _apply_preprocessing_and_augmentation(self, images: np.ndarray, preprocessing_config: dict, 
                                            augmentation_config: dict, apply_augmentation: bool = True) -> np.ndarray:
        """Apply preprocessing and augmentation to images."""
        import tensorflow as tf
        
        processed_images = images.copy()
        
        try:
            # Convert to TensorFlow tensors for processing
            tf_images = tf.constant(processed_images, dtype=tf.float32)
            
            # Apply preprocessing
            print("  🔧 Applying preprocessing...")
            
            # Resizing
            resizing_config = preprocessing_config.get('Resizing', {})
            if resizing_config.get('enabled', False):
                target_size = resizing_config.get('target_size', {})
                if target_size:
                    height = target_size.get('height', 224)
                    width = target_size.get('width', 224)
                    print(f"    📐 Resizing to {width}x{height}")
                    tf_images = tf.image.resize(tf_images, [height, width])
            
            # Normalization
            normalization_config = preprocessing_config.get('Normalization', {})
            if normalization_config.get('enabled', False):
                method = normalization_config.get('method', 'zero-center')
                print(f"    🔢 Applying {method} normalization")
                
                if method == 'min-max':
                    min_val = normalization_config.get('min_value', 0.0)
                    max_val = normalization_config.get('max_value', 1.0)
                    tf_images = tf_images * (max_val - min_val) + min_val
                elif method == 'zero-center':
                    # Apply ImageNet statistics if available
                    mean = [normalization_config.get('mean', {}).get(c, 0.5) for c in ['r', 'g', 'b']]
                    std = [normalization_config.get('std', {}).get(c, 0.5) for c in ['r', 'g', 'b']]
                    tf_images = (tf_images - mean) / std
            
            # Apply augmentations only for training split
            if apply_augmentation:
                print("  🎨 Applying augmentations...")
                
                # Horizontal flip
                hflip_config = augmentation_config.get('Horizontal Flip', {})
                if hflip_config.get('enabled', False):
                    prob = hflip_config.get('probability', 0.5)
                    if np.random.random() < prob:
                        print(f"    🔄 Horizontal flip (prob: {prob})")
                        tf_images = tf.image.flip_left_right(tf_images)
                
                # Vertical flip
                vflip_config = augmentation_config.get('Vertical Flip', {})
                if vflip_config.get('enabled', False):
                    prob = vflip_config.get('probability', 0.5)
                    if np.random.random() < prob:
                        print(f"    🔄 Vertical flip (prob: {prob})")
                        tf_images = tf.image.flip_up_down(tf_images)
                
                # Rotation
                rotation_config = augmentation_config.get('Rotation', {})
                if rotation_config.get('enabled', False):
                    prob = rotation_config.get('probability', 0.5)
                    if np.random.random() < prob:
                        angle_range = rotation_config.get('angle_range', 15.0)
                        angle = np.random.uniform(-angle_range, angle_range) * np.pi / 180
                        print(f"    🔄 Rotation by {angle * 180 / np.pi:.1f}° (prob: {prob})")
                        tf_images = self._rotate_images(tf_images, angle)
                
                # Brightness
                brightness_config = augmentation_config.get('Brightness', {})
                if brightness_config.get('enabled', False):
                    prob = brightness_config.get('probability', 0.5)
                    if np.random.random() < prob:
                        delta_range = brightness_config.get('delta_range', 0.2)
                        delta = np.random.uniform(-delta_range, delta_range)
                        print(f"    💡 Brightness adjustment: {delta:+.2f} (prob: {prob})")
                        tf_images = tf.image.adjust_brightness(tf_images, delta)
                
                # Contrast
                contrast_config = augmentation_config.get('Contrast', {})
                if contrast_config.get('enabled', False):
                    prob = contrast_config.get('probability', 0.5)
                    if np.random.random() < prob:
                        factor_range = contrast_config.get('factor_range', [0.8, 1.2])
                        factor = np.random.uniform(factor_range[0], factor_range[1])
                        print(f"    🌈 Contrast adjustment: {factor:.2f}x (prob: {prob})")
                        tf_images = tf.image.adjust_contrast(tf_images, factor)
                
                # Gaussian Noise
                noise_config = augmentation_config.get('Gaussian Noise', {})
                if noise_config.get('enabled', False):
                    prob = noise_config.get('probability', 0.5)
                    if np.random.random() < prob:
                        std_dev = noise_config.get('std_dev', 0.1)
                        print(f"    🎲 Gaussian noise: std={std_dev} (prob: {prob})")
                        noise = tf.random.normal(shape=tf.shape(tf_images), stddev=std_dev)
                        tf_images = tf_images + noise
            
            # Convert back to numpy and ensure valid range
            processed_images = tf_images.numpy()
            
            return processed_images
            
        except Exception as e:
            print(f"⚠️ Error applying preprocessing/augmentation: {str(e)}")
            return images  # Return original images if processing fails

    def _rotate_images(self, images, angle):
        """Rotate images by the given angle in radians."""
        import tensorflow as tf
        
        # Simple rotation using tf.contrib.image.rotate if available, 
        # otherwise use identity (no rotation)
        try:
            # For newer TensorFlow versions
            rotated = tf.keras.utils.image_utils.apply_affine_transform(
                images, theta=angle, fill_mode='nearest'
            )
            return rotated
        except:
            # Fallback: return original images if rotation not available
            print(f"    ⚠️ Rotation not available, skipping...")
            return images


def create_main_argument_parser():
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        description="ModelGardener CLI - Train and manage deep learning models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  config      Modify existing model configuration files
  train       Train a model
  evaluate    Evaluate a trained model
  predict     Run prediction on new data
  deploy      Deploy model in multiple formats (ONNX, TFLite, TF.js, etc.)
  models      List available models
  create      Create a new project template
  check       Check configuration files
  preview     Preview data samples with preprocessing and augmentation visualization

Examples:
  # Create a new project with interactive setup
  modelgardener_cli.py create my_project --interactive
  modelgardener_cli.py create my_project --dir /path/to/workspace
  modelgardener_cli.py create --interactive  # Create in current directory
  modelgardener_cli.py create  # Create basic template in current directory
  
  # Modify existing configuration files
  modelgardener_cli.py config config.yaml --interactive
  modelgardener_cli.py config --interactive  # Auto-finds config in current dir
  modelgardener_cli.py config config.yaml --epochs 100 --learning-rate 0.01
  
  # Check configuration files
  modelgardener_cli.py check config.yaml
  modelgardener_cli.py check config.json --verbose
  
  # Preview data samples
  modelgardener_cli.py preview --config config.yaml
  modelgardener_cli.py preview --config config.yaml --num-samples 12 --split val
  modelgardener_cli.py preview --config config.yaml --save --output data_samples.png
  modelgardener_cli.py preview --config config.yaml --split train --num-samples 16
  
  # Train a model
  modelgardener_cli.py train --config config.yaml
  
  # Evaluate the trained model
  modelgardener_cli.py evaluate --config config.yaml --data-path ./test_data
  modelgardener_cli.py evaluate --config config.yaml --output-format json
  
  # Run predictions
  modelgardener_cli.py predict --config config.yaml --input image.jpg
  modelgardener_cli.py predict --config config.yaml --input ./images/ --output results.json
  
  # Deploy models
  modelgardener_cli.py deploy --config config.yaml --formats onnx tflite
  modelgardener_cli.py deploy --config config.yaml --formats onnx --quantize --encrypt --encryption-key mykey
  
  # List available models
  modelgardener_cli.py models
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Modify existing model configuration')
    config_parser.add_argument('config_file', nargs='?', help='Existing configuration file to modify (optional - will search current directory)')
    config_parser.add_argument('--interactive', '-i', action='store_true', help='Interactive configuration modification mode')
    config_parser.add_argument('--format', '-f', choices=['json', 'yaml'], help='Output format (inferred from file extension if not specified)')
    
    # Add configuration modification arguments for batch mode
    config_parser.add_argument('--train-dir', type=str, help='Training data directory')
    config_parser.add_argument('--val-dir', type=str, help='Validation data directory')
    config_parser.add_argument('--batch-size', type=int, help='Batch size')
    config_parser.add_argument('--model-family', help='Model family')
    config_parser.add_argument('--model-name', type=str, help='Model name')
    config_parser.add_argument('--num-classes', type=int, help='Number of classes')
    config_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    config_parser.add_argument('--learning-rate', type=float, help='Learning rate')
    config_parser.add_argument('--optimizer', help='Optimizer')
    config_parser.add_argument('--loss-function', help='Loss function')
    config_parser.add_argument('--model-dir', type=str, help='Model output directory')
    config_parser.add_argument('--num-gpus', type=int, help='Number of GPUs to use')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--config', '-c', type=str, required=True, help='Configuration file')
    train_parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    train_parser.add_argument('--checkpoint', type=str, help='Checkpoint file to resume from')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--config', '-c', type=str, required=True, help='Configuration file')
    eval_parser.add_argument('--model-path', type=str, help='Path to trained model')
    eval_parser.add_argument('--data-path', type=str, help='Path to evaluation data')
    eval_parser.add_argument('--output-format', choices=['yaml', 'json'], default='yaml', help='Output format for results')
    eval_parser.add_argument('--no-save', action='store_true', help='Do not save evaluation results')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Run prediction on new data')
    predict_parser.add_argument('--config', '-c', type=str, required=True, help='Configuration file')
    predict_parser.add_argument('--input', '-i', type=str, required=True, help='Input image file or directory')
    predict_parser.add_argument('--model-path', type=str, help='Path to trained model')
    predict_parser.add_argument('--output', '-o', type=str, help='Output file for results (JSON/YAML)')
    predict_parser.add_argument('--top-k', type=int, default=5, help='Number of top predictions to show')
    predict_parser.add_argument('--batch-size', type=int, default=32, help='Batch size for processing')
    
    # Deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy model in multiple formats')
    deploy_parser.add_argument('--config', '-c', type=str, required=True, help='Configuration file')
    deploy_parser.add_argument('--model-path', type=str, help='Path to trained model')
    deploy_parser.add_argument('--formats', nargs='+', choices=['onnx', 'tflite', 'tfjs', 'keras'], 
                              default=['onnx', 'tflite'], help='Output formats')
    deploy_parser.add_argument('--quantize', action='store_true', help='Apply quantization (ONNX/TFLite)')
    deploy_parser.add_argument('--encrypt', action='store_true', help='Encrypt model files')
    deploy_parser.add_argument('--encryption-key', type=str, help='Encryption key for model files')
    
    # Models command
    models_parser = subparsers.add_parser('models', help='List available models')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create new project template')
    create_parser.add_argument('project_name', nargs='?', default=None, help='Name of the project (optional - uses current directory name if not provided)')
    create_parser.add_argument('--dir', '-d', default='.', help='Directory to create project in (ignored if no project_name provided)')
    create_parser.add_argument('--interactive', '-i', action='store_true', help='Interactive project creation mode')
    # Add configuration arguments for batch mode (same as config)
    create_parser.add_argument('--train-dir', type=str, help='Training data directory')
    create_parser.add_argument('--val-dir', type=str, help='Validation data directory')
    create_parser.add_argument('--batch-size', type=int, help='Batch size')
    create_parser.add_argument('--model-family', help='Model family')
    create_parser.add_argument('--model-name', type=str, help='Model name')
    create_parser.add_argument('--num-classes', type=int, help='Number of classes')
    create_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    create_parser.add_argument('--learning-rate', type=float, help='Learning rate')
    create_parser.add_argument('--optimizer', help='Optimizer')
    create_parser.add_argument('--loss-function', help='Loss function')
    create_parser.add_argument('--model-dir', type=str, help='Model output directory')
    create_parser.add_argument('--num-gpus', type=int, help='Number of GPUs to use')
    
    # Check command
    check_parser = subparsers.add_parser('check', help='Check configuration file')
    check_parser.add_argument('config_file', help='Configuration file to check')
    check_parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed validation results')
    
    # Preview command
    preview_parser = subparsers.add_parser('preview', help='Preview data samples with preprocessing and augmentation visualization')
    preview_parser.add_argument('--config', '-c', type=str, required=True, help='Configuration file')
    preview_parser.add_argument('--num-samples', '-n', type=int, default=8, help='Number of samples to preview (default: 8)')
    preview_parser.add_argument('--split', '-s', choices=['train', 'val', 'test'], default='train', help='Data split to preview (default: train)')
    preview_parser.add_argument('--save', action='store_true', help='Save plot to file instead of displaying')
    preview_parser.add_argument('--output', '-o', type=str, help='Output file path for saved plot')
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_main_argument_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = ModelGardenerCLI()
    
    try:
        if args.command == 'config':
            # Handle configuration command - only modify existing configs
            config_file = args.config_file
            
            # If no config file specified, try to find one in current directory
            if not config_file:
                config_file = cli.find_config_file(".")
                if not config_file:
                    print("❌ No configuration file found in current directory.")
                    print("💡 Expected files: config.yaml, config.yml, model_config.yaml, model_config.yml, config.json, or model_config.json")
                    print("💡 Use 'create' command to create a new project with configuration.")
                    return
                else:
                    print(f"🔍 Found configuration file: {config_file}")
            
            # Prepare kwargs for batch mode
            kwargs = {}
            if hasattr(args, 'train_dir') and args.train_dir:
                kwargs['train_dir'] = args.train_dir
            if hasattr(args, 'val_dir') and args.val_dir:
                kwargs['val_dir'] = args.val_dir
            if hasattr(args, 'batch_size') and args.batch_size:
                kwargs['batch_size'] = args.batch_size
            if hasattr(args, 'model_family') and args.model_family:
                kwargs['model_family'] = args.model_family
            if hasattr(args, 'model_name') and args.model_name:
                kwargs['model_name'] = args.model_name
            if hasattr(args, 'num_classes') and args.num_classes:
                kwargs['num_classes'] = args.num_classes
            if hasattr(args, 'epochs') and args.epochs:
                kwargs['epochs'] = args.epochs
            if hasattr(args, 'learning_rate') and args.learning_rate:
                kwargs['learning_rate'] = args.learning_rate
            if hasattr(args, 'optimizer') and args.optimizer:
                kwargs['optimizer'] = args.optimizer
            if hasattr(args, 'loss_function') and args.loss_function:
                kwargs['loss_function'] = args.loss_function
            if hasattr(args, 'model_dir') and args.model_dir:
                kwargs['model_dir'] = args.model_dir
            if hasattr(args, 'num_gpus') and args.num_gpus is not None:
                kwargs['num_gpus'] = args.num_gpus
            
            # Modify the existing configuration
            success = cli.modify_existing_config(config_file, args.interactive, **kwargs)
            if not success:
                sys.exit(1)
        
        elif args.command == 'train':
            success = cli.run_training(args.config)
            sys.exit(0 if success else 1)
        
        elif args.command == 'evaluate':
            # Handle new evaluation parameters
            data_path = getattr(args, 'data_path', None)
            output_format = getattr(args, 'output_format', 'yaml')
            save_results = not getattr(args, 'no_save', False)
            
            success = cli.run_evaluation(args.config, args.model_path, data_path, output_format, save_results)
            sys.exit(0 if success else 1)
        
        elif args.command == 'predict':
            success = cli.run_prediction(args.config, args.input, args.model_path, 
                                       getattr(args, 'output', None), args.top_k, args.batch_size)
            sys.exit(0 if success else 1)
        
        elif args.command == 'deploy':
            success = cli.run_deployment(args.config, args.model_path, args.formats,
                                       args.quantize, args.encrypt, getattr(args, 'encryption_key', None))
            sys.exit(0 if success else 1)
        
        elif args.command == 'models':
            cli.list_available_models()
        
        elif args.command == 'create':
            # Prepare kwargs for batch mode
            kwargs = {}
            if hasattr(args, 'train_dir') and args.train_dir:
                kwargs['train_dir'] = args.train_dir
            if hasattr(args, 'val_dir') and args.val_dir:
                kwargs['val_dir'] = args.val_dir
            if hasattr(args, 'batch_size') and args.batch_size:
                kwargs['batch_size'] = args.batch_size
            if hasattr(args, 'model_family') and args.model_family:
                kwargs['model_family'] = args.model_family
            if hasattr(args, 'model_name') and args.model_name:
                kwargs['model_name'] = args.model_name
            if hasattr(args, 'num_classes') and args.num_classes:
                kwargs['num_classes'] = args.num_classes
            if hasattr(args, 'epochs') and args.epochs:
                kwargs['epochs'] = args.epochs
            if hasattr(args, 'learning_rate') and args.learning_rate:
                kwargs['learning_rate'] = args.learning_rate
            if hasattr(args, 'optimizer') and args.optimizer:
                kwargs['optimizer'] = args.optimizer
            if hasattr(args, 'loss_function') and args.loss_function:
                kwargs['loss_function'] = args.loss_function
            if hasattr(args, 'model_dir') and args.model_dir:
                kwargs['model_dir'] = args.model_dir
            if hasattr(args, 'num_gpus') and args.num_gpus is not None:
                kwargs['num_gpus'] = args.num_gpus
            cli.create_project_template(args.project_name, args.dir, args.interactive, **kwargs)
        
        elif args.command == 'check':
            success = cli.check_configuration(args.config_file, args.verbose)
            sys.exit(0 if success else 1)
        
        elif args.command == 'preview':
            success = cli.preview_data(args.config, args.num_samples, args.split, 
                                     args.save, getattr(args, 'output', None))
            sys.exit(0 if success else 1)
    
    except KeyboardInterrupt:
        print("\n\n⚡ Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
