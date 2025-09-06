"""
ModelGardener CLI entry point for 'python -m modelgardener'
"""

from .cli import run_cli

def main():
    """Main entry point for the CLI when run as a module."""
    run_cli()

if __name__ == "__main__":
    main()
