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
    from cli_config import ModelConfigCLI, create_argument_parser
    from config_manager import ConfigManager
    from enhanced_trainer import EnhancedTrainer
except ImportError as e:
    print(f"❌ Error importing required modules: {e}")
    print("💡 Make sure you're running from the ModelGardener directory")
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
            
            # Initialize trainer
            trainer = EnhancedTrainer(
                config=main_config,
                custom_functions=config.get('metadata', {}).get('custom_functions', {})
            )
            
            # Run training
            print("\n🏃 Starting training...")
            success = trainer.train()
            
            if success:
                print("✅ Training completed successfully!")
                return True
            else:
                print("❌ Training failed")
                return False
                
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
            
            # Use RefactoredEnhancedTrainer for consistent behavior
            from refactored_enhanced_trainer import RefactoredEnhancedTrainer
            trainer = RefactoredEnhancedTrainer(
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
            
            # Use RefactoredEnhancedTrainer for consistent behavior
            from refactored_enhanced_trainer import RefactoredEnhancedTrainer
            trainer = RefactoredEnhancedTrainer(
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
            
            # Use RefactoredEnhancedTrainer for consistent behavior
            from refactored_enhanced_trainer import RefactoredEnhancedTrainer
            trainer = RefactoredEnhancedTrainer(
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
    
    except KeyboardInterrupt:
        print("\n\n⚡ Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
