"""
Core CLI functionality for ModelGardener
"""

import sys
import os
from pathlib import Path

def setup_module_path():
    """Add the root directory to Python path for imports."""
    # Get the directory containing this package
    root_dir = Path(__file__).parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

def run_cli():
    """Run the CLI application."""
    setup_module_path()
    
    try:
        # Import and run the CLI main function
        from modelgardener_cli import main as cli_main
        cli_main()
    except ImportError as e:
        print(f"❌ Error importing CLI modules: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install inquirer==3.1.3")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running CLI: {e}")
        sys.exit(1)
