"""
ModelGardener CLI Entry Point

This module provides the main entry point for the ModelGardener CLI application.
"""

import sys
from pathlib import Path


def main():
    """Main entry point for ModelGardener CLI."""
    try:
        # Import and run CLI interface
        from .modelgardener_cli from . import main as cli_main
        
        # Run CLI main function
        cli_main()
    except ImportError as e:
        print(f"❌ Error importing CLI modules: {e}")
        print("💡 Make sure all CLI dependencies are installed:")
        print("   uv add inquirer")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running CLI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
