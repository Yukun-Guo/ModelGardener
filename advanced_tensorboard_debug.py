#!/usr/bin/env python3
"""
Advanced TensorBoard debugging - check actual file system and configuration
"""

import os
import glob
from pathlib import Path


def check_tensorboard_files():
    """Check for actual TensorBoard event files in the filesystem"""
    
    print("🔍 Checking for TensorBoard event files in the filesystem...\n")
    
    # Common paths where TensorBoard logs might be
    search_paths = [
        './logs/tensorboard',
        './model_dir', 
        './model_dir/tensorboard',
        './logs',
        '.',
        './test_model_dir',
        './test_model_dir/tensorboard'
    ]
    
    found_any = False
    
    for search_path in search_paths:
        if os.path.exists(search_path):
            print(f"📁 Checking directory: {search_path}")
            
            # Look for TensorBoard event files
            event_files = []
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    if 'tfevents' in file or file.endswith('.v2'):
                        event_files.append(os.path.join(root, file))
            
            if event_files:
                print(f"  ✅ Found {len(event_files)} TensorBoard event files:")
                for event_file in event_files:
                    file_size = os.path.getsize(event_file)
                    mod_time = os.path.getmtime(event_file)
                    from datetime import datetime
                    mod_time_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
                    print(f"    - {event_file} ({file_size} bytes, modified: {mod_time_str})")
                found_any = True
            else:
                print(f"  ❌ No TensorBoard event files found")
                
            # Check if directory has any subdirectories
            subdirs = [d for d in os.listdir(search_path) if os.path.isdir(os.path.join(search_path, d))]
            if subdirs:
                print(f"  📂 Subdirectories: {subdirs}")
        else:
            print(f"📁 Directory does not exist: {search_path}")
        print()
    
    if not found_any:
        print("❌ No TensorBoard event files found anywhere!")
        print("   This means either:")
        print("   1. Training hasn't been run yet")
        print("   2. TensorBoard callback is not being created") 
        print("   3. TensorBoard callback is failing to write files")
        print("   4. Files are being written to an unexpected location")
    
    return found_any


def check_current_configuration():
    """Check the current configuration that would be used"""
    
    print("🔧 Checking current configuration...\n")
    
    # Try to load the current GUI configuration
    try:
        import sys
        sys.path.append('.')
        
        # Try to access main window if it exists
        print("📋 Configuration check:")
        print("  - Default TensorBoard log_dir from callbacks_group.py: ./logs/tensorboard")
        print("  - Default model_dir from runtime: ./model_dir") 
        print("  - Expected TensorBoard path: ./logs/tensorboard")
        print()
        
        # Check if these directories exist
        if os.path.exists('./logs'):
            print("✅ ./logs directory exists")
        else:
            print("❌ ./logs directory does not exist")
            
        if os.path.exists('./logs/tensorboard'):
            print("✅ ./logs/tensorboard directory exists")
        else:
            print("❌ ./logs/tensorboard directory does not exist")
            
        if os.path.exists('./model_dir'):
            print("✅ ./model_dir directory exists") 
        else:
            print("❌ ./model_dir directory does not exist")
        print()
        
    except Exception as e:
        print(f"Could not check configuration: {e}")
        print()


def create_test_tensorboard_structure():
    """Create the expected directory structure for testing"""
    
    print("🛠️  Creating test TensorBoard directory structure...\n")
    
    try:
        # Create the expected directories
        os.makedirs('./logs/tensorboard', exist_ok=True)
        print("✅ Created ./logs/tensorboard directory")
        
        # Create a simple test event file to see if TensorBoard can read it
        import tempfile
        import time
        
        test_file_path = f"./logs/tensorboard/events.out.tfevents.{int(time.time())}.test"
        with open(test_file_path, 'wb') as f:
            # Write minimal TensorBoard event file header
            f.write(b'\x18\x00\x00\x00\x00\x00\x00\x00')  # Minimal event file
        
        print(f"✅ Created test event file: {test_file_path}")
        print("   Now try refreshing TensorBoard to see if it detects this directory")
        
    except Exception as e:
        print(f"❌ Error creating test structure: {e}")


def main():
    print("🚀 TensorBoard 'No dashboards are active' - Advanced Debug\n")
    
    # Check for existing event files
    found_files = check_tensorboard_files()
    
    # Check current configuration
    check_current_configuration()
    
    # If no files found, offer to create test structure
    if not found_files:
        create_test_tensorboard_structure()
        print()
        print("💡 Next steps:")
        print("1. Start training to see if TensorBoard callback creates event files")
        print("2. Check the training logs for 'Added TensorBoard callback' message")
        print("3. If callback is created but no files appear, there may be a callback error")
        print("4. Check TensorBoard server startup log for the correct directory path")
    else:
        print("💡 Files were found! The issue might be:")
        print("1. TensorBoard server is looking in wrong directory")
        print("2. TensorBoard server needs to be restarted") 
        print("3. Browser cache needs to be refreshed")


if __name__ == '__main__':
    main()
