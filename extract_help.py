#!/usr/bin/env python3
"""
Extract help information from ModelGardener CLI for documentation updates.
"""

import sys
import os
import argparse

# Mock the necessary modules to avoid tensorflow import issues
class MockModule:
    def __getattr__(self, name):
        return MockModule()
    
    def __call__(self, *args, **kwargs):
        return MockModule()

# Mock tensorflow and other heavy dependencies
sys.modules['tensorflow'] = MockModule()
sys.modules['tensorflow.keras'] = MockModule()
sys.modules['tensorflow.python'] = MockModule()
sys.modules['tensorflow.python.tools'] = MockModule()
sys.modules['keras'] = MockModule()

# Add src to path
sys.path.insert(0, '/mnt/sda1/WorkSpace/ModelGardener/src')

def extract_help_info():
    """Extract help information from the CLI."""
    try:
        from modelgardener.modelgardener_cli import create_main_argument_parser
        
        parser = create_main_argument_parser()
        
        print("=" * 80)
        print("MAIN HELP")
        print("=" * 80)
        parser.print_help()
        
        # Get subparsers
        subparsers_actions = [
            action for action in parser._actions 
            if isinstance(action, argparse._SubParsersAction)
        ]
        
        if subparsers_actions:
            for subparsers_action in subparsers_actions:
                for choice, subparser in subparsers_action.choices.items():
                    print(f"\n{'=' * 80}")
                    print(f"{choice.upper()} COMMAND HELP")
                    print("=" * 80)
                    subparser.print_help()
        
    except Exception as e:
        print(f"Error extracting help: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    extract_help_info()
