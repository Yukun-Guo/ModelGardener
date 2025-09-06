"""
Core CLI functionality for ModelGardener
"""

import sys


def run_cli():
    """Run the CLI application."""
    try:
        # Import and run the CLI main function
        from .modelgardener_cli import main as cli_main
        cli_main()
    except ImportError as e:
        print(f"❌ Error importing CLI modules: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   uv add inquirer tensorflow")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running CLI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_cli()
