import sys
from pathlib import Path

if __name__ == "__main__":
    # CLI Mode Only - Import and run CLI interface
    try:
        # Add current directory to path for CLI imports
        sys.path.insert(0, str(Path(__file__).parent))
        from modelgardener_cli_bk import main as cli_main
        
        # Run CLI main function
        cli_main()
    except ImportError as e:
        print(f"❌ Error importing CLI modules: {e}")
        print("💡 Make sure all CLI dependencies are installed:")
        print("   pip install inquirer==3.1.3")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running CLI: {e}")
        sys.exit(1)
