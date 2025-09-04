#!/usr/bin/env python3
"""
ModelGardener CLI - Main entry point using modular configuration.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CLIInterface
from script_generator import ScriptGenerator


def create_argument_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description='ModelGardener CLI - ML model configuration and training setup',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s create my_project                    # Create project with default config
  %(prog)s create my_project -i                # Interactive configuration
  %(prog)s create my_project --config my.yaml  # Use existing config file
  %(prog)s config --help                       # Show configuration options
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create a new ML project')
    create_parser.add_argument('project_name', help='Name of the project to create')
    create_parser.add_argument('-i', '--interactive', action='store_true',
                              help='Use interactive configuration mode')
    create_parser.add_argument('-c', '--config', type=str,
                              help='Path to existing configuration file')
    create_parser.add_argument('-o', '--output-dir', type=str, default='.',
                              help='Output directory for the project (default: current directory)')
    create_parser.add_argument('--format', choices=['yaml', 'json'], default='yaml',
                              help='Configuration file format (default: yaml)')
    
    # Config command  
    config_parser = subparsers.add_parser('config', help='Configuration utilities')
    config_parser.add_argument('--validate', type=str,
                              help='Validate a configuration file')
    config_parser.add_argument('--convert', nargs=2, metavar=('INPUT', 'OUTPUT'),
                              help='Convert configuration between formats')
    config_parser.add_argument('--template', action='store_true',
                              help='Generate a template configuration file')
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        # Initialize CLI interface
        cli = CLIInterface()
        
        if args.command == 'create':
            create_project(cli, args)
        elif args.command == 'config':
            handle_config_command(cli, args)
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def create_project(cli: CLIInterface, args):
    """Create a new ML project."""
    project_name = args.project_name
    output_dir = Path(args.output_dir)
    project_dir = output_dir / project_name
    
    print(f"🌱 Creating ModelGardener project: {project_name}")
    print(f"📁 Project directory: {project_dir}")
    
    # Create project directory
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # Get configuration
    if args.interactive:
        print("\n🔧 Interactive project configuration")
        print("=" * 50)
        config = cli.interactive_configuration()
        if not config:
            print("❌ Configuration failed")
            return
    elif args.config:
        print(f"📄 Loading configuration from: {args.config}")
        config = cli.load_config(args.config)
        if not config:
            print("❌ Failed to load configuration file")
            return
    else:
        print("📋 Using default configuration")
        config = cli.create_default_config()
    
    # Save configuration
    config_file = project_dir / f"config.{args.format}"
    success = cli.save_config(config, str(config_file), args.format)
    if not success:
        print("❌ Failed to save configuration")
        return
    
    print(f"\n{cli.generate_config_summary(config)}")
    
    # Generate Python scripts
    print("\n🐍 Generating Python scripts...")
    try:
        script_gen = ScriptGenerator()
        success = script_gen.generate_scripts(config, str(project_dir), str(config_file))
        if success:
            cli.print_success("Python scripts generated successfully!")
        else:
            cli.print_warning("Some scripts may not have been generated correctly")
        
        print(f"📁 Location: {project_dir.absolute()}")
        print("📄 Generated files:")
        
        generated_files = [
            ("train.py", "Training script"),
            ("evaluation.py", "Evaluation script"),
            ("prediction.py", "Prediction script"),
            ("deploy.py", "Deployment script"),
            ("requirements.txt", "Python dependencies"),
            ("README.md", "Usage instructions"),
            ("custom_modules/", "Custom function templates")
        ]
        
        for filename, description in generated_files:
            file_path = project_dir / filename
            if file_path.exists():
                print(f"   • {filename} - {description}")
        
    except (ImportError, AttributeError, OSError) as e:
        cli.print_error(f"Failed to generate scripts: {e}")
        return
    
    cli.print_success(f"Configuration saved to {config_file}")
    cli.print_success("Project template created successfully!")
    print(f"📖 See {project_dir}/README.md for instructions")


def handle_config_command(cli: CLIInterface, args):
    """Handle configuration utility commands."""
    if args.validate:
        print(f"🔍 Validating configuration: {args.validate}")
        config = cli.load_config(args.validate)
        if config:
            cli.print_success("Configuration is valid")
            print(cli.generate_config_summary(config))
        else:
            cli.print_error("Configuration validation failed")
    
    elif args.convert:
        input_file, output_file = args.convert
        print(f"🔄 Converting {input_file} → {output_file}")
        
        config = cli.load_config(input_file)
        if not config:
            cli.print_error("Failed to load input configuration")
            return
        
        # Determine output format from extension
        output_format = 'yaml' if output_file.endswith(('.yaml', '.yml')) else 'json'
        success = cli.save_config(config, output_file, output_format)
        
        if success:
            cli.print_success("Configuration converted successfully")
        else:
            cli.print_error("Conversion failed")
    
    elif args.template:
        print("📝 Generating template configuration...")
        config = cli.create_default_config()
        
        template_file = "template_config.yaml"
        success = cli.save_config(config, template_file)
        
        if success:
            cli.print_success(f"Template configuration saved to {template_file}")
        else:
            cli.print_error("Failed to generate template")
    
    else:
        print("❓ Please specify a configuration command. Use --help for options.")


if __name__ == "__main__":
    main()
