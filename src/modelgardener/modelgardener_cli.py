#!/usr/bin/env python3
"""
ModelGardener CLI Entry Point
Provides command-line access to ModelGardener functionality without the GUI.
"""

import argparse
import sys
import os
import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import yaml
import json
import numpy as np
from PIL import Image
import tensorflow as tf

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
    
    def _find_default_config(self) -> Optional[str]:
        """Find default configuration file in current directory."""
        current_dir = Path(".")
        config_files = ["config.yaml", "config.yml", "model_config.yaml", "model_config.yml"]
        
        for config_file in config_files:
            config_path = current_dir / config_file
            if config_path.exists():
                return str(config_path)
        return None
    
    def _find_latest_model(self) -> Optional[str]:
        """Find the latest model in logs/ directory structure."""
        logs_dir = Path("logs")
        
        if not logs_dir.exists():
            return None
        
        # Look for subdirectories in logs/
        subdirs = [d for d in logs_dir.iterdir() if d.is_dir()]
        
        if not subdirs:
            return None
        
        # Sort subdirectories by modification time (latest first)
        subdirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Look for final_model.keras in each subdirectory
        for subdir in subdirs:
            model_files = ["final_model.keras", "model.keras", "best_model.keras"]
            for model_file in model_files:
                model_path = subdir / model_file
                if model_path.exists():
                    return str(model_path)
        
        return None
    
    def _find_default_input(self) -> Optional[str]:
        """Find default input directory or file for prediction."""
        current_dir = Path(".")
        
        # Look for common test/prediction directories (expanded list)
        test_dirs = [
            "test", "test_data", "test_images", "prediction_data", 
            "val", "validation", "data/test", "data/val", "data/validation",
            "data/test_data", "data/test_images", "eval", "evaluate"
        ]
        
        for test_dir in test_dirs:
            test_path = current_dir / test_dir
            if test_path.exists() and test_path.is_dir():
                # Check if directory contains images or is a proper dataset directory
                image_extensions = ["*.jpg", "*.png", "*.jpeg", "*.bmp", "*.tiff", "*.webp"]
                image_files = []
                for ext in image_extensions:
                    image_files.extend(list(test_path.glob(ext)))
                    # Also check subdirectories (for class-based organization)
                    image_files.extend(list(test_path.glob(f"*/{ext}")))
                
                if image_files:
                    return str(test_path)
                
                # Check if it contains subdirectories that look like class directories
                subdirs = [d for d in test_path.iterdir() if d.is_dir()]
                if subdirs:
                    # Check if any subdirectory contains images
                    for subdir in subdirs:
                        for ext in image_extensions:
                            if list(subdir.glob(ext)):
                                return str(test_path)
        
        # Look for single image files in current directory
        for ext in ["*.jpg", "*.png", "*.jpeg", "*.bmp", "*.tiff", "*.webp"]:
            image_files = list(current_dir.glob(ext))
            if image_files:
                # Return the first image file found
                return str(image_files[0])
        
        return None

    def _generate_prediction_output_path(self, input_path: str) -> str:
        """Generate automatic output path for predictions."""
        if os.path.isfile(input_path):
            # Single file prediction
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            output_dir = "predictions"
        else:
            # Directory prediction
            base_name = os.path.basename(input_path.rstrip('/'))
            output_dir = "predictions"
        
        # Create predictions directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamped filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{base_name}_predictions_{timestamp}.json"
        
        return os.path.join(output_dir, output_file)

    def _save_prediction_report(self, results: list, config_file: str, model_path: str, 
                               input_path: str, output_path: str):
        """Save comprehensive prediction report with metadata."""
        try:
            # Create predictions directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else "predictions", exist_ok=True)
            
            # Build comprehensive report
            report = {
                "metadata": {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "config_file": os.path.abspath(config_file),
                    "model_path": os.path.abspath(model_path),
                    "input_path": os.path.abspath(input_path),
                    "total_predictions": len(results)
                },
                "predictions": results,
                "summary": {
                    "processed_files": len(results),
                    "successful_predictions": len([r for r in results if 'predictions' in r]),
                    "failed_predictions": len([r for r in results if 'error' in r])
                }
            }
            
            # Save as JSON
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"📊 Comprehensive prediction report saved to: {output_path}")
            
            # Also save simplified CSV if there are multiple predictions
            if len(results) > 1:
                csv_path = output_path.replace('.json', '.csv')
                self._save_prediction_csv(results, csv_path)
                print(f"📋 CSV summary saved to: {csv_path}")
                
        except Exception as e:
            print(f"⚠️ Warning: Could not save prediction report: {e}")

    def _save_prediction_csv(self, results: list, csv_path: str):
        """Save prediction results as CSV for easy analysis."""
        try:
            import pandas as pd
            
            # Flatten results for CSV
            csv_data = []
            for result in results:
                if 'predictions' in result:
                    file_path = result.get('file_path', 'unknown')
                    for i, pred in enumerate(result['predictions']):
                        csv_data.append({
                            'file_path': file_path,
                            'rank': i + 1,
                            'class': pred.get('class', ''),
                            'confidence': pred.get('confidence', 0.0)
                        })
                elif 'error' in result:
                    csv_data.append({
                        'file_path': result.get('file_path', 'unknown'),
                        'rank': 1,
                        'class': 'ERROR',
                        'confidence': 0.0,
                        'error': result['error']
                    })
            
            if csv_data:
                df = pd.DataFrame(csv_data)
                df.to_csv(csv_path, index=False)
        except ImportError:
            # pandas not available, skip CSV generation
            pass
        except Exception as e:
            print(f"⚠️ Warning: Could not save CSV: {e}")

    def validate_cli_arguments(self, args: argparse.Namespace) -> bool:
        """Validate CLI arguments before processing."""
        errors = []
        
        # Validate epochs parameter
        if hasattr(args, 'epochs') and args.epochs is not None:
            if args.epochs < 1:
                errors.append(f"epochs must be at least 1, got {args.epochs}")
        
        # Validate learning rate parameter
        if hasattr(args, 'learning_rate') and args.learning_rate is not None:
            if args.learning_rate <= 0:
                errors.append(f"learning_rate must be positive, got {args.learning_rate}")

        # Validate batch size parameter
        if hasattr(args, 'batch_size') and args.batch_size is not None:
            if args.batch_size < 1:
                errors.append(f"batch_size must be at least 1, got {args.batch_size}")
        
        # Validate number of classes
        if hasattr(args, 'num_classes') and args.num_classes is not None:
            if args.num_classes < 1:
                errors.append(f"num_classes must be at least 1, got {args.num_classes}")

        # Validate input dimensions
        if hasattr(args, 'input_height') and args.input_height is not None:
            if args.input_height < 16 or args.input_height > 2048:
                errors.append(f"input_height must be between 16 and 2048, got {args.input_height}")
        
        if hasattr(args, 'input_width') and args.input_width is not None:
            if args.input_width < 16 or args.input_width > 2048:
                errors.append(f"input_width must be between 16 and 2048, got {args.input_width}")
        
        if hasattr(args, 'input_channels') and args.input_channels is not None:
            if args.input_channels < 1:
                errors.append(f"input_channels must be at least 1, got {args.input_channels}")

        # # Validate number of GPUs
        # if hasattr(args, 'num_gpus') and args.num_gpus is not None:
        #     if args.num_gpus < 0:
        #         errors.append(f"num_gpus must be at least 0, got {args.num_gpus}")

        # Validate top-k for prediction
        if hasattr(args, 'top_k') and args.top_k is not None:
            if args.top_k < 1 or args.top_k > 100:
                errors.append(f"top_k must be between 1 and 100, got {args.top_k}")
        
        # Validate num_samples for preview
        if hasattr(args, 'num_samples') and args.num_samples is not None:
            if args.num_samples < 1:
                errors.append(f"num_samples must be at least 1, got {args.num_samples}")
        
        # Print errors if any
        if errors:
            print("❌ Parameter validation failed:")
            for error in errors:
                print(f"   • {error}")
            return False
        
        return True

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

    def _check_for_generated_script(self, script_name: str, working_dir: str = ".") -> Optional[str]:
        """Check if a generated script exists in the working directory."""
        script_path = Path(working_dir) / f"{script_name}.py"
        if script_path.exists() and script_path.is_file():
            return str(script_path)
        return None
    
    def _run_generated_script(self, script_path: str, additional_args: List[str] = None) -> bool:
        """Run a generated Python script with optional additional arguments."""
        import subprocess
        import sys
        
        try:
            cmd = [sys.executable, script_path]
            if additional_args:
                cmd.extend(additional_args)
            
            print(f"🚀 Running generated script: {script_path}")
            result = subprocess.run(cmd, cwd=Path(script_path).parent)
            return result.returncode == 0
        except Exception as e:
            print(f"❌ Error running generated script: {e}")
            return False

    def run_training(self, config_file: str, **kwargs):
        """Run training using CLI configuration, preferring generated scripts."""
        print(f"🚀 Starting ModelGardener training")
        
        # Check for generated training script first
        working_dir = Path(config_file).parent if config_file != "config.yaml" else "."
        generated_script = self._check_for_generated_script("train", working_dir)
        
        if generated_script:
            print(f"🎯 Found generated training script: {generated_script}")
            print("📜 Running generated script instead of CLI training...")
            success = self._run_generated_script(generated_script)
            return success
        else:
            print("📄 No generated training script found, using CLI training procedure...")
        
        # Fallback to original CLI training
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
            
            # Set the config file path for copying to versioned directory
            trainer.set_config_file_path(os.path.abspath(config_file))
            
            # Run training
            return trainer.train()
                
        except Exception as e:
            print(f"❌ Error during training: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            return False

    def run_evaluation(self, config_file: str = None, model_path: str = None, data_path: str = None, 
                      output_format: str = "yaml", save_results: bool = True):
        """Run enhanced model evaluation using CLI, preferring generated scripts."""
        print(f"📊 Starting ModelGardener evaluation")
        
        # Auto-discover configuration file if not provided
        if not config_file:
            config_file = self._find_default_config()
            if not config_file:
                print("❌ No configuration file found. Please specify with -c or ensure config.yaml exists in current directory.")
                return False
            print(f"🔍 Using auto-discovered config: {config_file}")
        
        # Auto-discover model if not provided
        if not model_path:
            model_path = self._find_latest_model()
            if not model_path:
                print("❌ No model found in logs/ directory. Please specify with -m or train a model first.")
                return False
            print(f"🔍 Using auto-discovered model: {model_path}")
        
        # Check for generated evaluation script first
        working_dir = Path(config_file).parent if config_file != "config.yaml" else "."
        generated_script = self._check_for_generated_script("evaluation", working_dir)
        
        if generated_script:
            print(f"🎯 Found generated evaluation script: {generated_script}")
            print("📜 Running generated script instead of CLI evaluation...")
            
            # Build command line arguments for the generated script
            args = []
            if model_path:
                args.extend(["--model-path", model_path])
            if data_path:
                args.extend(["--data-path", data_path])
            if output_format:
                args.extend(["--output-format", output_format])
            if not save_results:
                args.append("--no-save")
            
            success = self._run_generated_script(generated_script, args)
            return success
        else:
            print("📄 No generated evaluation script found, using CLI evaluation procedure...")
        
        # Fallback to original CLI evaluation with enhanced features
        print(f"📄 Configuration: {config_file}")
        print(f"🤖 Model path: {model_path}")
        if data_path:
            print(f"📁 Custom data path: {data_path}")
        
        if not os.path.exists(config_file):
            print(f"❌ Configuration file not found: {config_file}")
            return False
        
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False
        
        try:
            # Load configuration
            config = self.config_cli.load_config(config_file)
            if not config:
                print("❌ Failed to load configuration")
                return False
            
            main_config = config.get('configuration', {})
            
            # Set the model path
            main_config['runtime']['model_path'] = model_path
            
            # Use provided data path for evaluation if specified
            if data_path:
                if not os.path.exists(data_path):
                    print(f"❌ Custom data path not found: {data_path}")
                    return False
                main_config['data']['test_dir'] = data_path
                print(f"📁 Using custom evaluation data: {data_path}")
            else:
                # Use evaluation dataset from config
                eval_data_dir = main_config.get('data', {}).get('val_dir', None)
                if eval_data_dir:
                    print(f"📁 Using evaluation data from config: {eval_data_dir}")
                else:
                    print("⚠️  No evaluation data specified in config or command line")
            
            print("✅ Configuration loaded for evaluation")
            
            # Use EnhancedTrainer for consistent behavior
            trainer = EnhancedTrainer(
                config=main_config,
                custom_functions=config.get('metadata', {}).get('custom_functions', {})
            )
            
            # Set the config file path for consistency
            trainer.set_config_file_path(os.path.abspath(config_file))
            
            # Follow the same pipeline as training for data loading
            print("🔧 Setting up evaluation pipeline...")
            
            # Phase 1: Setup runtime (needed for strategy and GPU configuration)
            success = trainer._setup_runtime()
            if not success:
                print("❌ Failed to setup runtime for evaluation")
                return False
            
            # Phase 2: Create data pipeline (this loads the evaluation dataset)
            success = trainer._create_data_pipeline()
            if not success:
                print("❌ Failed to create data pipeline for evaluation")
                return False
            
            # Load the specific model
            print(f"🔄 Loading model from: {model_path}")
            if model_path.endswith('.keras'):
                trainer.model = tf.keras.models.load_model(model_path)
            else:
                # Try loading as SavedModel format
                trainer.model = tf.keras.models.load_model(model_path)
            
            if not trainer.model:
                print("❌ Failed to load model")
                return False
            
            # Use validation dataset for evaluation if no custom data was specified
            eval_dataset = trainer.val_dataset if not data_path else trainer.val_dataset
            if eval_dataset is None:
                print("❌ No evaluation dataset available")
                print("💡 Hint: Make sure your config has 'val_dir' specified or use -d to specify evaluation data")
                return False
            
            # Run evaluation
            print("\n📈 Starting evaluation...")
            results = trainer.evaluate(eval_dataset)
            
            if results:
                print("✅ Evaluation completed successfully!")
                print("\n📊 Evaluation Results:")
                print("=" * 50)
                for metric, value in results.items():
                    if isinstance(value, float):
                        print(f"  {metric:.<30} {value:.4f}")
                    else:
                        print(f"  {metric:.<30} {value}")
                print("=" * 50)
                
                # Save results if requested
                if save_results:
                    self._save_evaluation_report(results, config_file, model_path, data_path, output_format)
                
                return True
            else:
                print("❌ Evaluation failed")
                return False
                
        except Exception as e:
            print(f"❌ Error during evaluation: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def run_prediction(self, config_file: str = None, input_path: str = None, model_path: str = None, 
                      output_path: str = None, top_k: int = 5, batch_size: int = 32, save_results: bool = True):
        """Run prediction using CLI, preferring generated scripts."""
        print(f"🔮 Starting ModelGardener prediction")
        
        # Auto-discover configuration file if not provided
        if not config_file:
            config_file = self._find_default_config()
            if not config_file:
                print("❌ No configuration file found. Please specify with -c or ensure config.yaml exists in current directory.")
                return False
            print(f"🔍 Using auto-discovered config: {config_file}")
        
        # Auto-discover model if not provided
        if not model_path:
            model_path = self._find_latest_model()
            if not model_path:
                print("❌ No model found in logs/ directory. Please specify with -m or train a model first.")
                return False
            print(f"🔍 Using auto-discovered model: {model_path}")
        
        # Auto-discover input if not provided
        if not input_path:
            input_path = self._find_default_input()
            if not input_path:
                print("❌ No input data found. Please specify with -i or ensure test data exists.")
                return False
            print(f"🔍 Using auto-discovered input: {input_path}")
        
        # Check for generated prediction script first
        working_dir = Path(config_file).parent if config_file != "config.yaml" else "."
        generated_script = self._check_for_generated_script("prediction", working_dir)
        
        if generated_script:
            print(f"🎯 Found generated prediction script: {generated_script}")
            print("📜 Running generated script instead of CLI prediction...")
            
            # Build command line arguments for the generated script
            args = ["--input", input_path]
            if model_path:
                args.extend(["--model-path", model_path])
            if output_path:
                args.extend(["--output", output_path])
            if top_k != 5:
                args.extend(["--top-k", str(top_k)])
            if batch_size != 32:
                args.extend(["--batch-size", str(batch_size)])
            if not save_results:
                args.append("--no-save")
            
            success = self._run_generated_script(generated_script, args)
            return success
        else:
            print("📄 No generated prediction script found, using CLI prediction procedure...")
        
        # Fallback to enhanced CLI prediction
        print(f"📄 Configuration: {config_file}")
        print(f"📁 Input: {input_path}")
        print(f"🤖 Model path: {model_path}")
        
        if not os.path.exists(config_file):
            print(f"❌ Configuration file not found: {config_file}")
            return False
        
        if not os.path.exists(input_path):
            print(f"❌ Input path not found: {input_path}")
            return False
        
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False
        
        try:
            # Load configuration
            config = self.config_cli.load_config(config_file)
            if not config:
                print("❌ Failed to load configuration")
                return False
            
            main_config = config.get('configuration', {})
            
            # Set the model path
            main_config['runtime']['model_path'] = model_path
            
            print("✅ Configuration loaded for prediction")
            
            # Use EnhancedTrainer for consistent behavior (same as evaluation)
            trainer = EnhancedTrainer(
                config=main_config,
                custom_functions=config.get('metadata', {}).get('custom_functions', {})
            )
            
            # Set the config file path for consistency
            trainer.set_config_file_path(os.path.abspath(config_file))
            
            # Follow the same pipeline as training/evaluation for consistency
            print("🔧 Setting up prediction pipeline...")
            
            # Phase 1: Setup runtime (needed for strategy and GPU configuration)
            success = trainer._setup_runtime()
            if not success:
                print("❌ Failed to setup runtime for prediction")
                return False
            
            # Load the specific model
            print(f"🔄 Loading model from: {model_path}")
            if model_path.endswith('.keras'):
                trainer.model = tf.keras.models.load_model(model_path)
            else:
                # Try loading as SavedModel format
                trainer.model = tf.keras.models.load_model(model_path)
            
            if not trainer.model:
                print("❌ Failed to load model")
                return False
            
            # Run prediction
            print(f"\n🔮 Starting prediction on {input_path}...")
            results = self._run_prediction_on_path(trainer, input_path, top_k, batch_size, main_config)
            
            if results:
                print("✅ Prediction completed successfully!")
                
                # Save results if requested
                if save_results:
                    if not output_path:
                        # Auto-generate output path
                        output_path = self._generate_prediction_output_path(input_path)
                    self._save_prediction_report(results, config_file, model_path, input_path, output_path)
                elif output_path:
                    # Save to specific path even if save_results is False (explicit user request)
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

    def run_deployment(self, config_file: str = None, model_path: str = None, output_formats: List[str] = None,
                      quantize: bool = False, encrypt: bool = False, encryption_key: str = None, output_dir: str = 'deployed_models'):
        """Run enhanced model deployment with multiple format support, preferring generated scripts."""
        print(f"🚀 Starting ModelGardener deployment")
        
        # Auto-discover configuration file if not provided
        if not config_file:
            config_file = self._find_default_config()
            if not config_file:
                print("❌ No configuration file found. Please specify with -c or ensure config.yaml exists in current directory.")
                return False
            print(f"🔍 Using auto-discovered config: {config_file}")
        
        # Auto-discover model if not provided
        if not model_path:
            model_path = self._find_latest_model()
            if not model_path:
                print("❌ No model found in logs/ directory. Please specify with -m or train a model first.")
                return False
            print(f"🔍 Using auto-discovered model: {model_path}")
        
        # Set default output formats if not provided
        if not output_formats:
            output_formats = ['onnx', 'tflite']
        
        # Check for generated deployment script first
        working_dir = Path(config_file).parent if config_file != "config.yaml" else "."
        generated_script = self._check_for_generated_script("deploy", working_dir)
        
        if generated_script:
            print(f"🎯 Found generated deployment script: {generated_script}")
            print("📜 Running generated script instead of CLI deployment...")
            
            # Build command line arguments for the generated script
            # Note: Generated script uses different argument format than CLI
            args = []
            if output_formats:
                # Check if conversion is needed
                if len(output_formats) > 1 or output_formats[0] != 'keras':
                    args.append("--convert")
                    args.extend(["--formats"] + output_formats)
                else:
                    # Single format deployment
                    args.extend(["--model-format", output_formats[0]])
            else:
                # Default behavior
                args.append("--convert")
                args.extend(["--formats", "onnx", "tflite"])
            
            if quantize:
                args.append("--quantize")
            if encrypt:
                args.append("--encrypt")
            if encryption_key:
                args.extend(["--encryption-key", encryption_key])
            
            # Note: Generated script doesn't use --model-path, it finds model automatically
            # Note: Generated script doesn't use --output-dir, it has its own output logic
            
            success = self._run_generated_script(generated_script, args)
            return success
        else:
            print("📄 No generated deployment script found, using CLI deployment procedure...")
        
        # Fallback to enhanced CLI deployment
        print(f"📄 Configuration: {config_file}")
        print(f"🤖 Model path: {model_path}")
        print(f"📦 Output formats: {', '.join(output_formats)}")
        print(f"� Output directory: {output_dir}")
        
        if not os.path.exists(config_file):
            print(f"❌ Configuration file not found: {config_file}")
            return False
        
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False
        
        try:
            # Load configuration
            config = self.config_cli.load_config(config_file)
            if not config:
                print("❌ Failed to load configuration")
                return False
            
            main_config = config.get('configuration', {})
            
            # Set the model path
            main_config['runtime']['model_path'] = model_path
            
            print("✅ Configuration loaded for deployment")
            
            # Use EnhancedTrainer for consistent behavior (same as evaluation/prediction)
            trainer = EnhancedTrainer(
                config=main_config,
                custom_functions=config.get('metadata', {}).get('custom_functions', {})
            )
            
            # Set the config file path for consistency
            trainer.set_config_file_path(os.path.abspath(config_file))
            
            # Follow the same pipeline as training/evaluation for consistency
            print("🔧 Setting up deployment pipeline...")
            
            # Phase 1: Setup runtime (needed for strategy and GPU configuration)
            success = trainer._setup_runtime()
            if not success:
                print("❌ Failed to setup runtime for deployment")
                return False
            
            # Load the specific model
            print(f"🔄 Loading model from: {model_path}")
            if model_path.endswith('.keras'):
                trainer.model = tf.keras.models.load_model(model_path)
            else:
                # Try loading as SavedModel format
                trainer.model = tf.keras.models.load_model(model_path)
            
            if not trainer.model:
                print("❌ Failed to load model")
                return False
            
            # Run deployment
            print(f"\n🚀 Starting model deployment to {output_dir}...")
            success = self._deploy_model_formats(trainer, main_config, output_formats, quantize, encrypt, encryption_key, output_dir)
            
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

    def _save_evaluation_report(self, results: Dict[str, float], config_file: str, model_path: str, 
                               data_path: str = None, output_format: str = "yaml"):
        """Save comprehensive evaluation report to evaluation/ folder."""
        try:
            # Create evaluation directory
            eval_dir = Path("evaluation")
            eval_dir.mkdir(exist_ok=True)
            
            # Create timestamp for unique report
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Prepare comprehensive report
            report = {
                "evaluation_summary": {
                    "timestamp": datetime.now().isoformat(),
                    "config_file": str(config_file),
                    "model_path": str(model_path),
                    "data_path": str(data_path) if data_path else "from_config",
                    "evaluation_metrics": results
                },
                "model_info": {
                    "model_file": str(Path(model_path).name),
                    "model_directory": str(Path(model_path).parent),
                    "model_size_mb": round(Path(model_path).stat().st_size / (1024 * 1024), 2) if Path(model_path).exists() else "unknown"
                },
                "detailed_metrics": {}
            }
            
            # Add detailed metrics breakdown
            for metric, value in results.items():
                if isinstance(value, (int, float)):
                    report["detailed_metrics"][metric] = {
                        "value": float(value),
                        "formatted": f"{value:.4f}",
                        "type": "numeric"
                    }
                else:
                    report["detailed_metrics"][metric] = {
                        "value": str(value),
                        "type": "text"
                    }
            
            # Save report with timestamp
            if output_format.lower() == 'json':
                report_file = eval_dir / f"evaluation_report_{timestamp}.json"
                with open(report_file, 'w') as f:
                    json.dump(report, f, indent=2)
            else:  # yaml
                report_file = eval_dir / f"evaluation_report_{timestamp}.yaml"
                with open(report_file, 'w') as f:
                    yaml.dump(report, f, default_flow_style=False, indent=2)
            
            # Also save a "latest" report for easy access
            latest_file = eval_dir / f"latest_evaluation.{output_format.lower()}"
            if output_format.lower() == 'json':
                with open(latest_file, 'w') as f:
                    json.dump(report, f, indent=2)
            else:
                with open(latest_file, 'w') as f:
                    yaml.dump(report, f, default_flow_style=False, indent=2)
            
            print(f"\n💾 Evaluation report saved:")
            print(f"   📄 Timestamped: {report_file}")
            print(f"   📄 Latest: {latest_file}")
            print(f"📁 Reports directory: {eval_dir.absolute()}")
            
        except Exception as e:
            print(f"⚠️ Could not save evaluation report: {str(e)}")
            # Still print results to console as fallback
            print("\n💾 Report could not be saved, but results are displayed above.")

    def _save_evaluation_results(self, results: Dict[str, float], model_dir: str, output_format: str):
        """Save evaluation results to file (legacy method)."""
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
                             quantize: bool, encrypt: bool, encryption_key: str, output_dir: str = 'deployed_models') -> bool:
        """Deploy model in multiple formats."""
        try:
            # Use provided output directory instead of config-based directory
            deploy_dir = os.path.abspath(output_dir)
            os.makedirs(deploy_dir, exist_ok=True)
            print(f"📁 Creating deployment directory: {deploy_dir}")
            
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

    def create_project_template(self, project_name: str = None, project_dir: str = ".", 
                               interactive: bool = False, auto_generate_scripts: bool = True,
                               use_pyproject: bool = True, verbose: bool = False, **kwargs):
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
        if verbose:
            print(f"📁 Project directory: {project_path}")

        # Create project structure
        (project_path / "data" / "train").mkdir(parents=True, exist_ok=True)
        (project_path / "data" / "val").mkdir(parents=True, exist_ok=True)
        (project_path / "logs").mkdir(exist_ok=True)
        (project_path / "custom_modules").mkdir(exist_ok=True)

        # Create configuration based on mode
        config_file = project_path / "config.yaml"
        config_data = None

        if interactive:
            print("\n🔧 Interactive project configuration")
            print("=" * 50)

            # Update paths to be relative to the project directory
            old_cwd = os.getcwd()
            os.chdir(project_path)

            try:
                # Check if we can actually run interactively
                if not self.config_cli.is_interactive:
                    print("⚠️  Non-interactive environment detected. Creating default configuration.")
                    config = self.config_cli.create_default_config()
                    # Set some reasonable defaults for common use cases
                    config['configuration']['data']['train_dir'] = './data/train'
                    config['configuration']['data']['val_dir'] = './data/val'
                    config['configuration']['model']['model_family'] = 'custom'
                    config['configuration']['model']['model_name'] = 'example_model'
                    config_data = config
                else:
                    config = self.config_cli.interactive_configuration()
                    config_data = config

                # Validate and save configuration using relative path
                config_file_relative = "config.yaml"
                if self.config_cli.validate_config(config):
                    self.config_cli.display_config_summary(config)
                    if self.config_cli.save_config(config, config_file_relative, 'yaml', generate_scripts=False):
                        if verbose:
                            print(f"✅ Configuration saved to {config_file}")
                else:
                    print("❌ Configuration validation failed, using default template")
                    self.config_cli.create_template(config_file_relative, 'yaml', verbose)
            finally:
                os.chdir(old_cwd)
        else:
            # Batch mode: create config using provided arguments or default template
            if kwargs:
                # Create a default config first, then apply batch modifications
                default_config = self.config_cli.create_default_config()
                config = self.config_cli.batch_configuration_with_existing(default_config, kwargs)
                config_data = config
                if self.config_cli.validate_config(config):
                    self.config_cli.display_config_summary(config)
                    if self.config_cli.save_config(config, str(config_file), 'yaml', generate_scripts=False):
                        if verbose:
                            print(f"✅ Configuration saved to {config_file}")
                    else:
                        print("❌ Failed to save configuration, using default template")
                        self.config_cli.create_template(str(config_file), 'yaml', verbose)
                else:
                    print("❌ Configuration validation failed, using default template")
                    self.config_cli.create_template(str(config_file), 'yaml', verbose)
            else:
                # No batch args, create default template
                config_data = None
                self.config_cli.create_template(str(config_file), 'yaml', verbose)
        
        # Always generate custom modules, regardless of script flag
        if verbose:
            print(f"\n� Generating custom modules...")
        try:
            from .script_generator import ScriptGenerator
            
            # Load config data if available
            if config_data is None and config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        import yaml
                        config_data = yaml.safe_load(f)
                except Exception as e:
                    if verbose:
                        print(f"⚠️ Could not load config for script generation: {e}")
                    config_data = {}
            
            generator = ScriptGenerator()
            
            # Always generate custom modules
            generator.generate_custom_modules_templates(str(project_path), verbose)
            
            # Generate main scripts only if requested
            if auto_generate_scripts:
                if verbose:
                    print(f"\n📜 Generating main training scripts...")
                success = generator.generate_scripts(
                    config_data=config_data or {},
                    output_dir=str(project_path),
                    config_file_name="config.yaml",
                    generate_pyproject=use_pyproject,
                    generate_requirements=not use_pyproject,
                    enable_main_scripts=auto_generate_scripts,
                    verbose=verbose
                )
                
                if success:
                    if verbose:
                        print("✅ Training scripts generated successfully!")
                else:
                    print("⚠️ Some scripts may not have been generated correctly")
            else:
                # Still need to generate pyproject/requirements even without main scripts
                if use_pyproject:
                    generator._generate_pyproject_toml(config_data.get('configuration', config_data) if config_data else {}, str(project_path), verbose)
                else:
                    generator._generate_requirements_txt(config_data.get('configuration', config_data) if config_data else {}, str(project_path), verbose)
                
                if verbose:
                    print("✅ Custom modules generated successfully! (Main scripts disabled)")
                    
        except ImportError as e:
            print(f"⚠️ Script generator not available: {e}")
        except Exception as e:
            print(f"⚠️ Error generating scripts: {e}")
        
        # Update README content based on what was actually generated
        package_file = "pyproject.toml" if use_pyproject else "requirements.txt"
        main_scripts_note = "auto-generated" if auto_generate_scripts else "can be generated with --script"
        custom_modules_note = "always auto-generated"
        
        # Create README
        readme_content = f"""# {project_name} - ModelGardener Project

## Project Structure
```
{project_name}/
├── data/
│   ├── train/          # Training data
│   └── val/            # Validation data
├── logs/               # Training logs and models
├── custom_modules/     # Custom functions ({custom_modules_note})
├── config.yaml         # Model configuration
├── train.py           # Training script ({main_scripts_note})
├── evaluation.py      # Evaluation script ({main_scripts_note})
├── prediction.py      # Prediction script ({main_scripts_note})
├── deploy.py          # Deployment script ({main_scripts_note})
├── {package_file}      # Python dependencies
└── README.md          # This file
```

## Quick Start

### 1. Prepare Your Data
Place your training images in `data/train/` and validation images in `data/val/`

### 2. Configure Your Model
Edit the `config.yaml` file to customize your model settings, or use the interactive configuration:
```bash
# Interactive configuration (overwrites config.yaml)
mg config --interactive --output config.yaml

# Or directly edit config.yaml
```

### 3. Install Dependencies
```bash
# If using pyproject.toml (recommended)
pip install -e .

# If using requirements.txt
pip install -r requirements.txt
```

### 4. Train Your Model
```bash
# Use the generated training script
python train.py

# Or use the CLI command
mg train --config config.yaml
```

### 5. Evaluate Your Model
```bash
# Use the generated evaluation script  
python evaluation.py

# Or use the CLI command
mg evaluate --config config.yaml --model-path logs/final_model.keras
```

### 6. Make Predictions
```bash
# Use the generated prediction script
python prediction.py --input path/to/image.jpg

# Or use the CLI command
mg predict --config config.yaml --input path/to/image.jpg
```

### 7. Deploy Your Model
```bash
# Use the generated deployment script
python deploy.py --port 5000

# Or use the CLI command
mg deploy --config config.yaml --port 5000
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
- Use interactive mode for guided configuration: `mg config --interactive`
- Check the custom_modules/README.md for detailed examples
- See the ModelGardener documentation for advanced usage
"""
        
        readme_file = project_path / "README.md"
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        
        print(f"✅ Project template created successfully!")
        if verbose:
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
            
            # Normalize and validate configuration
            normalized_config = self.config_cli.normalize_config_format(config)
            is_valid = self.config_cli.validate_config(normalized_config)
            
            if verbose and is_valid:
                # Display configuration summary with normalized config
                print("\n📋 Configuration Summary:")
                self.config_cli.display_config_summary(normalized_config)
            
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
                
                # Load images with robust error handling
                preview_images_list = []
                preview_labels = []  # Track labels only for successfully loaded images
                failed_loads = 0
                
                for i, img_path in enumerate(sample_files):
                    try:
                        from PIL import Image
                        img = Image.open(img_path)
                        img = img.convert('RGB')
                        img_array = np.array(img) / 255.0
                        preview_images_list.append(img_array)
                        preview_labels.append(sample_labels[i])  # Only add label for successful load
                    except Exception as e:
                        failed_loads += 1
                        print(f"⚠️ Failed to load image {img_path}: {e}")
                        continue
                
                # Report loading results
                total_attempts = len(sample_files)
                successful_loads = len(preview_images_list)
                print(f"📊 Image loading results: {successful_loads}/{total_attempts} successful")
                
                # If no images loaded successfully, try to generate placeholder data for testing
                if successful_loads == 0:
                    print("💡 No valid images found - generating placeholder data for preview")
                    print("🔧 This suggests your test data may contain invalid image files")
                    
                    # Generate placeholder data that looks like real images
                    import numpy as np
                    placeholder_images = []
                    placeholder_labels = []
                    
                    for i in range(min(num_samples, len(class_dirs))):
                        # Create a simple colorful placeholder image
                        img_size = (224, 224, 3)  # Standard RGB image size
                        
                        # Create different colored placeholder for each class
                        if i < len(class_dirs):
                            # Use different base colors for different classes
                            colors = [(1.0, 0.7, 0.7), (0.7, 1.0, 0.7), (0.7, 0.7, 1.0), 
                                     (1.0, 1.0, 0.7), (1.0, 0.7, 1.0), (0.7, 1.0, 1.0)]
                            base_color = colors[i % len(colors)]
                        else:
                            base_color = (0.8, 0.8, 0.8)  # Gray for extras
                        
                        # Create gradient pattern
                        placeholder = np.zeros(img_size)
                        for c in range(3):
                            for y in range(img_size[0]):
                                placeholder[y, :, c] = base_color[c] * (0.5 + 0.5 * y / img_size[0])
                        
                        # Add some simple pattern to make it more recognizable
                        center_y, center_x = img_size[0] // 2, img_size[1] // 2
                        for y in range(max(0, center_y - 20), min(img_size[0], center_y + 20)):
                            for x in range(max(0, center_x - 20), min(img_size[1], center_x + 20)):
                                if (y - center_y) ** 2 + (x - center_x) ** 2 <= 400:
                                    placeholder[y, x] = [0.9, 0.9, 0.9]  # White circle in center
                        
                        placeholder_images.append(placeholder)
                        placeholder_labels.append(i % len(class_dirs))
                    
                    preview_images_list = placeholder_images
                    preview_labels = placeholder_labels
                    print(f"✅ Generated {len(placeholder_images)} placeholder images for preview")
                
                preview_images_raw = np.array(preview_images_list) if preview_images_list else None
                class_names = class_dirs
                
            else:
                print(f"⚠️ Data loader '{selected_loader}' preview not yet implemented")
                print("💡 Currently supported: Custom_load_cifar10_npz_data, ImageDataLoader, Default")
                return False
            
            if preview_images_raw is None or len(preview_images_raw) == 0:
                print("❌ No images could be loaded or generated for preview")
                print("💡 Possible solutions:")
                print("   • Check that your data paths in the configuration are correct")
                print("   • Verify that image files are valid (not corrupted or fake)")
                print("   • Ensure the data directory contains actual image files")
                print("   • Try running 'mg check config.yaml --verbose' to validate your configuration")
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
  mg create my_project --interactive
  mg create my_project --dir /path/to/workspace
  mg create --interactive  # Create in current directory
  mg create  # Create basic template in current directory
  
  # Modify existing configuration files
  mg config config.yaml --interactive
  mg config --interactive  # Auto-finds config in current dir
  mg config config.yaml --epochs 100 --learning-rate 0.01
  
  # Check configuration files
  mg check config.yaml
  mg check config.json --verbose
  
  # Preview data samples
  mg preview --config config.yaml
  mg preview --config config.yaml --num-samples 12 --split val
  mg preview --config config.yaml --save --output data_samples.png
  mg preview --config config.yaml --split train --num-samples 16
  
  # Train a model
  mg train --config config.yaml
  
  # Evaluate the trained model (auto-discovery)
  mg evaluate  # Uses config.yaml and latest model in logs/
  mg evaluate -c config.yaml  # Specific config file
  mg evaluate -m logs/final_model.keras  # Specific model with default config
  mg evaluate -d ./test_data  # Specific evaluation data
  mg evaluate -c config.yaml -m logs/model.keras -d ./custom_test_data
  
  # Run predictions
  mg predict --config config.yaml --input image.jpg
  mg predict --config config.yaml --input ./images/ --output results.json
  
  # Deploy models
  mg deploy --config config.yaml --formats onnx tflite
  mg deploy --config config.yaml --formats onnx --quantize --encrypt --encryption-key mykey
  
  # List available models
  mg models
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
    train_parser.add_argument('--config', '-c', type=str, required=False, help='Configuration file (optional - will search for config.yaml in current directory if not provided)')
    train_parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    train_parser.add_argument('--checkpoint', type=str, help='Checkpoint file to resume from')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--config', '-c', type=str, help='Configuration file (optional - defaults to config.yaml in current directory)')
    eval_parser.add_argument('--model-path', '-m', type=str, help='Path to trained model (e.g., logs/final_model.keras)')
    eval_parser.add_argument('--data-path', '-d', type=str, help='Path to evaluation data directory')
    eval_parser.add_argument('--output-format', choices=['yaml', 'json'], default='yaml', help='Output format for results')
    eval_parser.add_argument('--no-save', action='store_true', help='Do not save evaluation results to evaluation/ folder')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Run prediction on new data')
    predict_parser.add_argument('--config', '-c', type=str, help='Configuration file (optional - defaults to config.yaml in current directory)')
    predict_parser.add_argument('--input', '-i', type=str, help='Input image file or directory (optional - auto-discovers test data)')
    predict_parser.add_argument('--model-path', '-m', type=str, help='Path to trained model (e.g., logs/final_model.keras)')
    predict_parser.add_argument('--output', '-o', type=str, help='Output file for results (JSON/YAML)')
    predict_parser.add_argument('--top-k', type=int, default=5, help='Number of top predictions to show')
    predict_parser.add_argument('--batch-size', type=int, default=32, help='Batch size for processing')
    predict_parser.add_argument('--no-save', action='store_true', help='Do not save prediction results to predictions/ folder')
    
    # Deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy model in multiple formats')
    deploy_parser.add_argument('--config', '-c', type=str, 
                              help='Configuration file (optional - defaults to config.yaml in current directory)')
    deploy_parser.add_argument('--model-path', '-m', type=str, 
                              help='Path to trained model (optional - auto-discovers latest model)')
    deploy_parser.add_argument('--formats', '-f', nargs='+', choices=['onnx', 'tflite', 'tfjs', 'keras'], 
                              default=['onnx', 'tflite'], help='Output formats')
    deploy_parser.add_argument('--quantize', '-q', action='store_true', help='Apply quantization (ONNX/TFLite)')
    deploy_parser.add_argument('--encrypt', '-e', action='store_true', help='Encrypt model files')
    deploy_parser.add_argument('--encryption-key', '-k', type=str, help='Encryption key for model files')
    deploy_parser.add_argument('--output-dir', '-o', type=str, default='deployed_models', 
                              help='Output directory for deployed models')
    
    # Models command
    models_parser = subparsers.add_parser('models', help='List available models')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create new project template')
    create_parser.add_argument('project_name', nargs='?', default=None, help='Name of the project (optional - uses current directory name if not provided)')
    create_parser.add_argument('--dir', '-d', default='.', help='Directory to create project in (ignored if no project_name provided)')
    create_parser.add_argument('--interactive', '-i', action='store_true', help='Interactive project creation mode')
    create_parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output during project creation')
    create_parser.add_argument('--script', action='store_true', default=True, help='Enable auto-generation of main training scripts (train.py, predict.py, evaluation.py, deploy.py) (default: True)')
    create_parser.add_argument('--no-script', action='store_false', dest='script', help='Disable auto-generation of main training scripts (custom_modules/ are always generated)')
    create_parser.add_argument('--use-pyproject', action='store_true', default=True, help='Generate pyproject.toml instead of requirements.txt (default: True)')
    create_parser.add_argument('--use-requirements', action='store_false', dest='use_pyproject', help='Generate requirements.txt instead of pyproject.toml')
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
    
    # Validate CLI arguments before processing
    if not cli.validate_cli_arguments(args):
        print("💡 Please check your parameter values and try again.")
        sys.exit(1)
    
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
            # Handle config file - if not specified, search current directory for config.yaml
            config_file = args.config
            
            if not config_file:
                # Try to find config.yaml specifically in current directory
                if os.path.exists("config.yaml"):
                    config_file = "config.yaml"
                    print(f"🔍 Found config.yaml in current directory, using: {config_file}")
                else:
                    print("❌ No config.yaml found in current directory")
                    print("💡 Either:")
                    print("   • Create a config.yaml file in the current directory")
                    print("   • Specify a config file: mg train --config path/to/config.yaml")
                    print("   • Use 'mg create' to create a new project with configuration")
                    sys.exit(1)
            
            success = cli.run_training(config_file)
            sys.exit(0 if success else 1)
        
        elif args.command == 'evaluate':
            # Handle new evaluation parameters with auto-discovery
            config_file = getattr(args, 'config', None)
            model_path = getattr(args, 'model_path', None)
            data_path = getattr(args, 'data_path', None)
            output_format = getattr(args, 'output_format', 'yaml')
            save_results = not getattr(args, 'no_save', False)
            
            success = cli.run_evaluation(config_file, model_path, data_path, output_format, save_results)
            sys.exit(0 if success else 1)
        
        elif args.command == 'predict':
            save_results = not getattr(args, 'no_save', False)
            success = cli.run_prediction(args.config, args.input, args.model_path, 
                                       getattr(args, 'output', None), args.top_k, args.batch_size, save_results)
            sys.exit(0 if success else 1)
        
        elif args.command == 'deploy':
            success = cli.run_deployment(args.config, args.model_path, args.formats,
                                       args.quantize, args.encrypt, getattr(args, 'encryption_key', None),
                                       getattr(args, 'output_dir', 'deployed_models'))
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
            
            # Get script generation options
            auto_generate_scripts = getattr(args, 'script', True)
            use_pyproject = getattr(args, 'use_pyproject', True)
            verbose = getattr(args, 'verbose', False)
            
            cli.create_project_template(
                args.project_name, args.dir, args.interactive, 
                auto_generate_scripts, use_pyproject, verbose, **kwargs
            )
        
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
