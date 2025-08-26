import sys
import argparse
from pathlib import Path

def create_cli_parser():
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="ModelGardener - Deep Learning Model Training Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch GUI (default)
  python main.py
  
  # Use CLI configuration tool
  python main.py --cli config --interactive
  
  # Train with CLI
  python main.py --cli train --config model_config.json
  
  # Show available models
  python main.py --cli models
        """
    )
    
    parser.add_argument('--cli', action='store_true', 
                       help='Use CLI interface instead of GUI')
    parser.add_argument('--experiment', type=str, default="image_classification_imagenet",
                       help='Experiment name for GUI mode')
    
    # For CLI mode, accept additional arguments that will be passed to the CLI
    parser.add_argument('cli_args', nargs='*', help='Arguments for CLI mode')
    
    return parser

if __name__ == "__main__":
    parser = create_cli_parser()
    args = parser.parse_args()
    
    if args.cli:
        # CLI Mode - Import and run CLI interface
        try:
            # Add current directory to path for CLI imports
            sys.path.insert(0, str(Path(__file__).parent))
            from modelgardener_cli import main as cli_main
            
            # Reconstruct sys.argv for CLI
            sys.argv = ['modelgardener_cli.py'] + args.cli_args
            cli_main()
        except ImportError as e:
            print(f"❌ Error importing CLI modules: {e}")
            print("💡 Make sure all CLI dependencies are installed:")
            print("   pip install inquirer==3.1.3")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error running CLI: {e}")
            sys.exit(1)
    else:
        # GUI Mode - Original functionality
        try:
            from PySide6.QtWidgets import QApplication
            from main_window import MainWindow
            
            app = QApplication(sys.argv)
            win = MainWindow(experiment_name=args.experiment)
            win.show()
            sys.exit(app.exec())
        except ImportError as e:
            print(f"❌ Error importing GUI modules: {e}")
            print("💡 GUI dependencies may not be installed or available.")
            print("💡 Consider using CLI mode: python main.py --cli")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error running GUI: {e}")
            print("💡 Consider using CLI mode as a fallback: python main.py --cli")
            sys.exit(1)
