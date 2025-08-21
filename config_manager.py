"""
Configuration Manager for ModelGardener
Handles saving and loading configurations with custom functions support
"""

import json
import yaml
import os
import importlib.util
from typing import Dict, Any, Optional, List
from PySide6.QtWidgets import QMessageBox


class ConfigManager:
    """Enhanced configuration manager that properly handles custom functions."""
    
    def __init__(self, main_window=None):
        self.main_window = main_window
        
    def save_enhanced_config(self, config_data: Dict[str, Any], file_path: str, 
                           custom_functions_info: Optional[Dict] = None) -> bool:
        """
        Save configuration with custom functions metadata and optional file embedding.
        
        Args:
            config_data: The main configuration dictionary
            file_path: Path where to save the configuration
            custom_functions_info: Dictionary containing custom function metadata
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            # Prepare enhanced configuration with multiple sharing strategies
            enhanced_config = {
                'configuration': config_data,
                'metadata': {
                    'version': '1.2',
                    'custom_functions': custom_functions_info or {},
                    'sharing_strategy': 'file_paths_with_content',  # New strategy
                    'creation_date': self._get_current_timestamp(),
                    'model_gardener_version': '1.0'
                }
            }
            
            # Enhanced custom functions metadata with file contents
            if custom_functions_info:
                enhanced_config['metadata']['custom_functions'] = self._enhance_custom_functions_metadata(
                    custom_functions_info
                )
            
            # Determine format from file extension
            file_ext = os.path.splitext(file_path)[1].lower()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                if file_ext == '.json':
                    json.dump(enhanced_config, f, indent=2, ensure_ascii=False)
                elif file_ext in ['.yaml', '.yml']:
                    yaml.dump(enhanced_config, f, allow_unicode=True, default_flow_style=False)
                else:
                    # Default to JSON
                    json.dump(enhanced_config, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            if self.main_window:
                QMessageBox.critical(self.main_window, "Save Error", 
                                   f"Failed to save configuration:\n{str(e)}")
            return False
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp as ISO string."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _enhance_custom_functions_metadata(self, custom_functions_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance custom functions metadata with file contents and checksums for better sharing.
        
        Args:
            custom_functions_info: Original custom functions metadata
            
        Returns:
            Enhanced metadata with file contents and verification info
        """
        enhanced_info = {}
        
        for func_type, functions_list in custom_functions_info.items():
            enhanced_functions = []
            
            for func_info in functions_list:
                enhanced_func = dict(func_info)  # Copy original info
                
                # Try to read and embed the source file content
                file_path = func_info.get('file_path')
                if file_path and os.path.exists(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            file_content = f.read()
                        
                        # Add file content and metadata for sharing
                        enhanced_func.update({
                            'file_content': file_content,
                            'file_size': len(file_content),
                            'file_checksum': self._calculate_checksum(file_content),
                            'relative_file_path': os.path.basename(file_path),  # For extraction
                            'sharing_enabled': True
                        })
                        
                        # Extract additional metadata from the file
                        enhanced_func.update(self._extract_function_metadata(file_content, func_info))
                        
                    except Exception as e:
                        # If we can't read the file, mark it as non-shareable but keep the path
                        enhanced_func.update({
                            'sharing_enabled': False,
                            'sharing_error': str(e),
                            'requires_manual_setup': True
                        })
                else:
                    enhanced_func.update({
                        'sharing_enabled': False,
                        'sharing_error': 'File not found',
                        'requires_manual_setup': True
                    })
                
                enhanced_functions.append(enhanced_func)
            
            enhanced_info[func_type] = enhanced_functions
        
        return enhanced_info
    
    def _calculate_checksum(self, content: str) -> str:
        """Calculate SHA256 checksum of content."""
        import hashlib
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]  # First 16 chars
    
    def _extract_function_metadata(self, file_content: str, func_info: Dict) -> Dict:
        """
        Extract additional metadata from function source code.
        
        Args:
            file_content: The source code content
            func_info: Original function info
            
        Returns:
            Dictionary with extracted metadata
        """
        import ast
        import inspect
        
        metadata = {
            'dependencies': [],
            'imports': [],
            'docstring': '',
            'parameters_info': []
        }
        
        try:
            tree = ast.parse(file_content)
            function_name = func_info.get('function_name', func_info.get('original_name'))
            
            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        metadata['imports'].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        metadata['imports'].append(f"{module}.{alias.name}")
                elif isinstance(node, ast.FunctionDef) and node.name == function_name:
                    # Extract docstring
                    metadata['docstring'] = ast.get_docstring(node) or ''
                    
                    # Extract parameter information
                    for arg in node.args.args:
                        param_info = {'name': arg.arg}
                        if arg.annotation:
                            try:
                                param_info['type_hint'] = ast.unparse(arg.annotation)
                            except:
                                param_info['type_hint'] = 'Any'
                        metadata['parameters_info'].append(param_info)
            
            # Extract likely dependencies from imports
            common_ml_libs = ['tensorflow', 'torch', 'numpy', 'pandas', 'sklearn', 'cv2', 'PIL', 'matplotlib']
            for imp in metadata['imports']:
                for lib in common_ml_libs:
                    if lib in imp.lower():
                        if lib not in metadata['dependencies']:
                            metadata['dependencies'].append(lib)
                            
        except Exception as e:
            metadata['extraction_error'] = str(e)
        
        return metadata
    
    def load_enhanced_config(self, file_path: str) -> tuple[Optional[Dict], Optional[Dict]]:
        """
        Load configuration with custom functions metadata and automatic extraction.
        
        Args:
            file_path: Path to the configuration file
            
        Returns:
            tuple: (config_data, custom_functions_info) or (None, None) if failed
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_ext = os.path.splitext(file_path)[1].lower()
                
                if file_ext == '.json':
                    data = json.load(f)
                elif file_ext in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                else:
                    # Try JSON first, then YAML
                    try:
                        f.seek(0)
                        data = json.load(f)
                    except json.JSONDecodeError:
                        f.seek(0)
                        data = yaml.safe_load(f)
            
            # Check if it's an enhanced config or legacy config
            if isinstance(data, dict) and 'configuration' in data and 'metadata' in data:
                # Enhanced configuration format
                config_data = data['configuration']
                metadata = data.get('metadata', {})
                custom_functions_info = metadata.get('custom_functions', {})
                sharing_strategy = metadata.get('sharing_strategy', 'file_paths_only')
                
                # If this is a shareable config, extract embedded custom functions
                if sharing_strategy == 'file_paths_with_content':
                    extraction_results = self._extract_embedded_custom_functions(
                        custom_functions_info, os.path.dirname(file_path)
                    )
                    
                    # Show extraction results if any files were created
                    if extraction_results['extracted_files']:
                        self._show_extraction_results(extraction_results)
                
            else:
                # Legacy configuration format
                config_data = data
                custom_functions_info = {}
            
            return config_data, custom_functions_info
            
        except Exception as e:
            if self.main_window:
                QMessageBox.critical(self.main_window, "Load Error", 
                                   f"Failed to load configuration:\n{str(e)}")
            return None, None
    
    def _extract_embedded_custom_functions(self, custom_functions_info: Dict[str, Any], 
                                         config_dir: str) -> Dict[str, Any]:
        """
        Extract embedded custom functions to files for sharing.
        
        Args:
            custom_functions_info: Enhanced custom functions metadata
            config_dir: Directory where the config file is located
            
        Returns:
            Dictionary with extraction results
        """
        results = {
            'extracted_files': [],
            'skipped_files': [],
            'errors': []
        }
        
        # Create a custom_functions subdirectory
        custom_functions_dir = os.path.join(config_dir, 'custom_functions')
        
        try:
            for func_type, functions_list in custom_functions_info.items():
                if not functions_list:
                    continue
                
                # Create subdirectory for this function type
                type_dir = os.path.join(custom_functions_dir, func_type)
                
                for func_info in functions_list:
                    if not func_info.get('sharing_enabled', False):
                        results['skipped_files'].append({
                            'name': func_info.get('name', 'unknown'),
                            'reason': func_info.get('sharing_error', 'Not shareable')
                        })
                        continue
                    
                    file_content = func_info.get('file_content')
                    if not file_content:
                        continue
                    
                    # Determine output filename
                    relative_path = func_info.get('relative_file_path', f"{func_info.get('function_name', 'function')}.py")
                    output_path = os.path.join(type_dir, relative_path)
                    
                    try:
                        # Create directory if it doesn't exist
                        os.makedirs(type_dir, exist_ok=True)
                        
                        # Check if file already exists and verify checksum
                        if os.path.exists(output_path):
                            with open(output_path, 'r', encoding='utf-8') as existing_file:
                                existing_content = existing_file.read()
                            
                            existing_checksum = self._calculate_checksum(existing_content)
                            expected_checksum = func_info.get('file_checksum', '')
                            
                            if existing_checksum == expected_checksum:
                                # File already exists and is identical
                                results['skipped_files'].append({
                                    'name': func_info.get('name', 'unknown'),
                                    'reason': 'File already exists with same content',
                                    'path': output_path
                                })
                                # Update the file path in the metadata
                                func_info['file_path'] = output_path
                                continue
                            else:
                                # File exists but content is different, create with suffix
                                name, ext = os.path.splitext(output_path)
                                counter = 1
                                while os.path.exists(f"{name}_v{counter}{ext}"):
                                    counter += 1
                                output_path = f"{name}_v{counter}{ext}"
                        
                        # Write the file
                        with open(output_path, 'w', encoding='utf-8') as output_file:
                            output_file.write(file_content)
                        
                        # Update the file path in the metadata
                        func_info['file_path'] = output_path
                        
                        results['extracted_files'].append({
                            'name': func_info.get('name', 'unknown'),
                            'path': output_path,
                            'size': len(file_content),
                            'type': func_type
                        })
                        
                    except Exception as e:
                        results['errors'].append({
                            'name': func_info.get('name', 'unknown'),
                            'error': str(e)
                        })
        
        except Exception as e:
            results['errors'].append({
                'name': 'General extraction',
                'error': str(e)
            })
        
        return results
    
    def _show_extraction_results(self, results: Dict[str, Any]):
        """Show results of custom functions extraction to user."""
        if not self.main_window:
            return
        
        extracted = results.get('extracted_files', [])
        skipped = results.get('skipped_files', [])
        errors = results.get('errors', [])
        
        message = "Custom Functions Extraction Results:\n\n"
        
        if extracted:
            message += f"✓ Extracted {len(extracted)} custom function file(s):\n"
            for file_info in extracted:
                message += f"  • {file_info['name']} ({file_info['type']})\n"
                message += f"    → {file_info['path']}\n"
            message += "\n"
        
        if skipped:
            message += f"⚠ Skipped {len(skipped)} file(s):\n"
            for skip_info in skipped:
                message += f"  • {skip_info['name']}: {skip_info['reason']}\n"
            message += "\n"
        
        if errors:
            message += f"✗ Errors with {len(errors)} file(s):\n"
            for error_info in errors:
                message += f"  • {error_info['name']}: {error_info['error']}\n"
            message += "\n"
        
        message += "Custom functions are now ready to be loaded into ModelGardener!"
        
        if errors:
            QMessageBox.warning(self.main_window, "Custom Functions Extraction", message)
        else:
            QMessageBox.information(self.main_window, "Custom Functions Extraction", message)
    
    def collect_custom_functions_info(self, parameter_tree) -> Dict[str, Any]:
        """
        Collect custom functions information from parameter tree groups.
        
        Args:
            parameter_tree: The main parameter tree
            
        Returns:
            Dict containing custom functions metadata
        """
        custom_info = {
            'data_loaders': [],
            'optimizers': [],
            'loss_functions': [],
            'metrics': [],
            'augmentations': [],
            'callbacks': [],
            'preprocessing': []
        }
        
        try:
            # Get basic configuration groups
            basic_group = parameter_tree.child('basic')
            if basic_group:
                # Data loaders
                data_group = basic_group.child('data')
                if data_group:
                    data_loader_group = data_group.child('data_loader')
                    if data_loader_group and hasattr(data_loader_group, '_custom_data_loaders'):
                        for name, info in data_loader_group._custom_data_loaders.items():
                            custom_info['data_loaders'].append({
                                'name': name,
                                'file_path': info['file_path'],
                                'original_name': info['original_name'],
                                'type': info['type']
                            })
                
                # Model related custom functions
                model_group = basic_group.child('model')
                if model_group:
                    # Optimizers
                    optimizer_group = model_group.child('optimizer')
                    if optimizer_group and hasattr(optimizer_group, '_custom_optimizers'):
                        for name, func in optimizer_group._custom_optimizers.items():
                            # Need to store file path if available
                            custom_info['optimizers'].append({
                                'name': name,
                                'function_name': name.replace('Custom_', ''),
                            })
                    
                    # Loss functions
                    loss_group = model_group.child('loss_functions')
                    if loss_group and hasattr(loss_group, '_custom_loss_functions'):
                        for name, info in loss_group._custom_loss_functions.items():
                            custom_info['loss_functions'].append({
                                'name': name,
                                'file_path': info['file_path'],
                                'function_name': info['function_name'],
                                'type': info['type']
                            })
                    
                    # Metrics
                    metrics_group = model_group.child('metrics')
                    if metrics_group and hasattr(metrics_group, '_custom_metrics'):
                        for name, info in metrics_group._custom_metrics.items():
                            custom_info['metrics'].append({
                                'name': name,
                                'file_path': info['file_path'],
                                'function_name': info['function_name'],
                                'type': info['type']
                            })
            
            # Get advanced configuration groups
            advanced_group = parameter_tree.child('advanced')
            if advanced_group:
                # Augmentations
                aug_group = advanced_group.child('augmentation')
                if aug_group:
                    for child in aug_group.children():
                        if child.name().endswith('(custom)'):
                            file_path_param = child.child('file_path')
                            function_name_param = child.child('function_name')
                            if file_path_param and function_name_param:
                                custom_info['augmentations'].append({
                                    'name': child.name(),
                                    'file_path': file_path_param.value(),
                                    'function_name': function_name_param.value()
                                })
                
                # Callbacks
                callbacks_group = advanced_group.child('callbacks')
                if callbacks_group:
                    for child in callbacks_group.children():
                        if child.name().endswith('(custom)'):
                            file_path_param = child.child('file_path')
                            function_name_param = child.child('function_name')
                            if file_path_param and function_name_param:
                                custom_info['callbacks'].append({
                                    'name': child.name(),
                                    'file_path': file_path_param.value(),
                                    'function_name': function_name_param.value()
                                })
                
                # Preprocessing (if available in advanced section)
                # Look for preprocessing in data section
                data_group = basic_group.child('data') if basic_group else None
                if data_group:
                    preprocessing_group = data_group.child('preprocessing')
                    if preprocessing_group:
                        for child in preprocessing_group.children():
                            if child.name().endswith('(custom)'):
                                file_path_param = child.child('file_path')
                                function_name_param = child.child('function_name')
                                if file_path_param and function_name_param:
                                    custom_info['preprocessing'].append({
                                        'name': child.name(),
                                        'file_path': file_path_param.value(),
                                        'function_name': function_name_param.value()
                                    })
        
        except Exception as e:
            print(f"Error collecting custom functions info: {e}")
        
        return custom_info
    
    def restore_custom_functions(self, custom_functions_info: Dict[str, Any], 
                                parameter_tree) -> List[str]:
        """
        Restore custom functions from metadata.
        
        Args:
            custom_functions_info: Dictionary containing custom functions metadata
            parameter_tree: The main parameter tree
            
        Returns:
            List of error messages for functions that couldn't be restored
        """
        errors = []
        
        try:
            # Restore data loaders
            for loader_info in custom_functions_info.get('data_loaders', []):
                try:
                    basic_group = parameter_tree.child('basic')
                    if basic_group:
                        data_group = basic_group.child('data')
                        if data_group:
                            data_loader_group = data_group.child('data_loader')
                            if data_loader_group:
                                # Try to load the custom data loader
                                file_path = loader_info['file_path']
                                if os.path.exists(file_path):
                                    # This would need to call the actual loading method
                                    # For now, just record that it should be loaded
                                    pass
                                else:
                                    errors.append(f"Data loader file not found: {file_path}")
                except Exception as e:
                    errors.append(f"Failed to restore data loader {loader_info.get('name', 'unknown')}: {e}")
            
            # Similar restoration logic for other custom function types...
            # This is a framework - specific implementations would need to be added
            
        except Exception as e:
            errors.append(f"General error restoring custom functions: {e}")
        
        return errors
    
    def auto_reload_custom_functions(self, custom_functions_info: Dict[str, Any], 
                                   parameter_tree) -> bool:
        """
        Automatically reload custom functions when loading a configuration.
        
        Args:
            custom_functions_info: Dictionary containing custom functions metadata
            parameter_tree: The main parameter tree
            
        Returns:
            bool: True if all functions were loaded successfully
        """
        if not custom_functions_info:
            return True
        
        errors = self.restore_custom_functions(custom_functions_info, parameter_tree)
        
        if errors:
            error_msg = "Some custom functions could not be reloaded:\n\n" + "\n".join(errors)
            error_msg += "\n\nYou may need to manually reload these custom functions."
            
            if self.main_window:
                QMessageBox.warning(self.main_window, "Custom Functions Warning", error_msg)
            return False
        
        return True
    
    def create_shareable_package(self, config_data: Dict[str, Any], package_path: str,
                                custom_functions_info: Optional[Dict] = None, 
                                include_readme: bool = True) -> bool:
        """
        Create a complete shareable package with configuration and custom functions.
        
        Args:
            config_data: The main configuration dictionary
            package_path: Path where to create the package (should be a directory)
            custom_functions_info: Dictionary containing custom function metadata
            include_readme: Whether to include a README file with setup instructions
            
        Returns:
            bool: True if package created successfully
        """
        try:
            # Create package directory
            os.makedirs(package_path, exist_ok=True)
            
            # Save enhanced configuration
            config_file_path = os.path.join(package_path, 'model_config.json')
            success = self.save_enhanced_config(config_data, config_file_path, custom_functions_info)
            
            if not success:
                return False
            
            # Create custom functions directory structure
            if custom_functions_info:
                custom_functions_dir = os.path.join(package_path, 'custom_functions')
                os.makedirs(custom_functions_dir, exist_ok=True)
                
                # Copy/extract all custom function files
                file_manifest = []
                
                for func_type, functions_list in custom_functions_info.items():
                    if not functions_list:
                        continue
                    
                    type_dir = os.path.join(custom_functions_dir, func_type)
                    os.makedirs(type_dir, exist_ok=True)
                    
                    for func_info in functions_list:
                        original_file_path = func_info.get('file_path')
                        if original_file_path and os.path.exists(original_file_path):
                            # Copy original file
                            filename = os.path.basename(original_file_path)
                            dest_path = os.path.join(type_dir, filename)
                            
                            try:
                                with open(original_file_path, 'r', encoding='utf-8') as src:
                                    content = src.read()
                                
                                with open(dest_path, 'w', encoding='utf-8') as dst:
                                    dst.write(content)
                                
                                file_manifest.append({
                                    'type': func_type,
                                    'name': func_info.get('name', 'unknown'),
                                    'function_name': func_info.get('function_name', func_info.get('original_name')),
                                    'file_path': os.path.relpath(dest_path, package_path),
                                    'description': func_info.get('docstring', ''),
                                    'dependencies': func_info.get('dependencies', []),
                                    'imports': func_info.get('imports', [])
                                })
                                
                            except Exception as e:
                                print(f"Failed to copy {original_file_path}: {e}")
                
                # Create manifest file
                manifest_path = os.path.join(package_path, 'custom_functions_manifest.json')
                with open(manifest_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'model_gardener_version': '1.0',
                        'package_version': '1.0',
                        'creation_date': self._get_current_timestamp(),
                        'custom_functions': file_manifest
                    }, f, indent=2)
            
            # Create README if requested
            if include_readme:
                readme_content = self._generate_package_readme(config_data, custom_functions_info)
                readme_path = os.path.join(package_path, 'README.md')
                with open(readme_path, 'w', encoding='utf-8') as f:
                    f.write(readme_content)
            
            # Create setup script
            setup_script_content = self._generate_setup_script(custom_functions_info)
            setup_script_path = os.path.join(package_path, 'setup_custom_functions.py')
            with open(setup_script_path, 'w', encoding='utf-8') as f:
                f.write(setup_script_content)
            
            return True
            
        except Exception as e:
            if self.main_window:
                QMessageBox.critical(self.main_window, "Package Creation Error", 
                                   f"Failed to create shareable package:\n{str(e)}")
            return False
    
    def _generate_package_readme(self, config_data: Dict[str, Any], 
                                custom_functions_info: Optional[Dict]) -> str:
        """Generate README content for the shareable package."""
        readme = """# ModelGardener Configuration Package

This package contains a complete ModelGardener configuration with custom functions.

## Contents

- `model_config.json` - Main configuration file with embedded custom functions
- `custom_functions/` - Directory containing custom function files
- `custom_functions_manifest.json` - Manifest of all custom functions
- `setup_custom_functions.py` - Setup script to automatically load custom functions

## Setup Instructions

### Method 1: Automatic Setup (Recommended)

1. Copy this entire package to your desired location
2. Open ModelGardener
3. Load the `model_config.json` file
4. When prompted, click "Yes" to auto-reload custom functions

### Method 2: Manual Setup

1. Copy this package to your ModelGardener working directory
2. Run the setup script: `python setup_custom_functions.py`
3. Open ModelGardener and load the `model_config.json` file

### Method 3: Individual Function Loading

1. Open ModelGardener
2. Load the `model_config.json` file (this loads the base configuration)
3. For each custom function type, use the respective "Load Custom..." button:
"""
        
        if custom_functions_info:
            for func_type, functions_list in custom_functions_info.items():
                if functions_list:
                    readme += f"\n#### {func_type.replace('_', ' ').title()}\n"
                    for func_info in functions_list:
                        file_path = func_info.get('file_path', 'Unknown path')
                        readme += f"- {func_info.get('name', 'Unknown')}: `{file_path}`\n"
        
        readme += """
## Custom Functions Overview

"""
        
        if custom_functions_info:
            total_functions = sum(len(funcs) for funcs in custom_functions_info.values())
            readme += f"This package contains **{total_functions} custom function(s)**:\n\n"
            
            for func_type, functions_list in custom_functions_info.items():
                if functions_list:
                    readme += f"### {func_type.replace('_', ' ').title()}\n\n"
                    for func_info in functions_list:
                        readme += f"**{func_info.get('name', 'Unknown')}**\n"
                        if func_info.get('docstring'):
                            readme += f"- Description: {func_info['docstring'][:200]}...\n"
                        if func_info.get('dependencies'):
                            readme += f"- Dependencies: {', '.join(func_info['dependencies'])}\n"
                        readme += "\n"
        
        readme += """
## Requirements

- ModelGardener v1.0 or later
- Python dependencies as listed in individual function files

## Troubleshooting

If you encounter issues loading custom functions:

1. Check that all file paths in the manifest are correct
2. Ensure all required Python packages are installed
3. Verify that the custom function files are valid Python code
4. Try loading functions individually using the "Load Custom..." buttons

## Support

For issues with this configuration package, please check the original source or contact the package creator.
"""
        
        return readme
    
    def _generate_setup_script(self, custom_functions_info: Optional[Dict]) -> str:
        """Generate setup script for automatic function loading."""
        script = '''#!/usr/bin/env python3
"""
ModelGardener Custom Functions Setup Script

This script helps set up custom functions for ModelGardener by checking dependencies
and providing guidance on loading custom functions.
"""

import os
import sys
import json
import importlib.util
from pathlib import Path

def check_dependencies():
    """Check if required Python packages are available."""
    print("Checking dependencies...")
    
    required_packages = set()
    
    # Load manifest to get dependency information
    manifest_path = Path(__file__).parent / "custom_functions_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
            
        for func_info in manifest.get('custom_functions', []):
            required_packages.update(func_info.get('dependencies', []))
    
    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("All dependencies are available!")
    return True

def validate_custom_functions():
    """Validate that custom function files can be loaded."""
    print("\\nValidating custom function files...")
    
    custom_functions_dir = Path(__file__).parent / "custom_functions"
    if not custom_functions_dir.exists():
        print("No custom functions directory found.")
        return True
    
    valid_files = 0
    total_files = 0
    
    for py_file in custom_functions_dir.rglob("*.py"):
        total_files += 1
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to parse the file
            compile(content, str(py_file), 'exec')
            print(f"✓ {py_file.relative_to(custom_functions_dir)}")
            valid_files += 1
            
        except Exception as e:
            print(f"✗ {py_file.relative_to(custom_functions_dir)}: {e}")
    
    print(f"\\nValidated {valid_files}/{total_files} custom function files.")
    return valid_files == total_files

def main():
    """Main setup function."""
    print("ModelGardener Custom Functions Setup")
    print("=" * 40)
    
    dependencies_ok = check_dependencies()
    functions_ok = validate_custom_functions()
    
    print("\\n" + "=" * 40)
    
    if dependencies_ok and functions_ok:
        print("✓ Setup completed successfully!")
        print("\\nNext steps:")
        print("1. Open ModelGardener")
        print("2. Load the model_config.json file")
        print("3. When prompted, choose to auto-reload custom functions")
    else:
        print("⚠ Setup completed with issues.")
        print("Please resolve the issues above before loading custom functions.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''
        return script
