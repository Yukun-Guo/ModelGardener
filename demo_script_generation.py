#!/usr/bin/env python3
"""
Demo script showing the new Python script generation feature.
Run this to see how ModelGardener now automatically generates 
ready-to-run Python scripts from YAML configurations.
"""

import os
import sys
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def run_demo():
    """Demonstrate the script generation feature."""
    
    print("🎉 ModelGardener Script Generation Demo")
    print("=" * 60)
    
    print("""
✨ NEW FEATURE: Automatic Python Script Generation ✨

When you save any configuration (YAML or JSON), ModelGardener now automatically 
generates ready-to-run Python scripts in the same directory:

📄 Generated Scripts:
• train.py        - Standalone training script
• evaluation.py   - Model evaluation script  
• prediction.py   - Prediction script for new images
• deploy.py       - REST API deployment script
• requirements.txt - Python dependencies
• README.md       - Usage instructions

🚀 These scripts are:
✅ Self-contained - Run independently with your configuration
✅ Customizable   - Generated code can be modified for your needs
✅ Production-ready - Include error handling, logging, best practices
✅ Cross-validation ready - Support k-fold CV when enabled
✅ API deployment ready - REST API with health checks
✅ Batch processing ready - Handle single images or directories
""")
    
    print("📋 Demo Commands:")
    print()
    print("1. Generate a sample configuration with scripts:")
    print("   python modelgardener_cli.py config --template --format yaml --output demo/config.yaml")
    print()
    print("2. Create interactive configuration:")
    print("   python modelgardener_cli.py config --interactive --output my_project/config.yaml")
    print()
    print("3. Use the generated scripts:")
    print("   cd my_project")
    print("   pip install -r requirements.txt")
    print("   python train.py                                    # Train model")
    print("   python evaluation.py                               # Evaluate model")
    print("   python prediction.py --input path/to/image.jpg    # Make predictions")
    print("   python deploy.py --port 8080                       # Deploy API")
    print()
    
    print("💡 The GUI also generates scripts when you save configurations!")
    print()
    
    # Ask user if they want to run a demo
    response = input("Would you like to run a quick demo? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        print("\n🎬 Running demo...")
        demo_dir = "demo_script_generation"
        
        try:
            # Create demo directory
            os.makedirs(demo_dir, exist_ok=True)
            
            # Run CLI command to generate template
            import subprocess
            cmd = [sys.executable, "modelgardener_cli.py", "config", "--template", 
                   "--format", "yaml", "--output", f"{demo_dir}/demo_config.yaml"]
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Demo completed successfully!")
                print(f"\n📁 Check the '{demo_dir}' directory to see the generated files:")
                
                if os.path.exists(demo_dir):
                    for file in sorted(os.listdir(demo_dir)):
                        file_path = os.path.join(demo_dir, file)
                        if os.path.isfile(file_path):
                            size = os.path.getsize(file_path)
                            print(f"   • {file} ({size} bytes)")
                
                print(f"\n💡 You can now:")
                print(f"   cd {demo_dir}")
                print(f"   pip install -r requirements.txt")  
                print(f"   python train.py")
            else:
                print("❌ Demo failed:")
                print(result.stderr)
                
        except Exception as e:
            print(f"❌ Demo error: {e}")
    
    print("\n🔗 For more information, see:")
    print("   • CLI_README.md - Complete CLI documentation")
    print("   • README.md - Project overview and features")

if __name__ == "__main__":
    run_demo()
