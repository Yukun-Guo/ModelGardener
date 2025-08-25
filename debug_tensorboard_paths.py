#!/usr/bin/env python3
"""
Test script to debug TensorBoard path resolution issue
"""

import os


def test_tensorboard_path_resolution():
    """Test the exact path resolution logic used in both files"""
    
    print("🔍 Testing TensorBoard path resolution logic...\n")
    
    # Simulate the GUI configuration
    gui_cfg = {
        'callbacks': {
            'TensorBoard': {
                'enabled': True,
                'log_dir': './logs/tensorboard'  # This is the GUI default
            }
        },
        'runtime': {
            'model_dir': './model_dir'  # This is the runtime default
        }
    }
    
    print("📋 Configuration:")
    print(f"  TensorBoard log_dir from GUI: {gui_cfg['callbacks']['TensorBoard']['log_dir']}")
    print(f"  Runtime model_dir: {gui_cfg['runtime']['model_dir']}")
    print()
    
    # Test main_window.py path resolution logic
    print("🏠 Main Window TensorBoard startup path resolution:")
    callbacks_cfg = gui_cfg.get("callbacks", {})
    tensorboard_cfg = callbacks_cfg.get("TensorBoard", {})
    
    if tensorboard_cfg.get('enabled', True):
        log_dir = tensorboard_cfg.get('log_dir', './logs/tensorboard')
        print(f"  1. Initial log_dir from config: {log_dir}")
        
        # Current logic from main_window.py
        runtime_cfg = gui_cfg.get("runtime", {})
        model_dir = runtime_cfg.get("model_dir", "./model_dir")  # Fixed default
        print(f"  2. Model dir: {model_dir}")
        
        if not os.path.isabs(log_dir) and not log_dir.startswith('./'):
            log_dir = os.path.join(model_dir, log_dir)
            print(f"  3. Path was modified to: {log_dir}")
        else:
            print(f"  3. Path kept as-is: {log_dir}")
        
        print(f"  ➤ Final TensorBoard startup path: {log_dir}")
    print()
    
    # Test enhanced_trainer.py path resolution logic  
    print("🤖 Enhanced Trainer callback path resolution:")
    callbacks_config = gui_cfg.get('callbacks', {})
    runtime_config = gui_cfg.get('runtime', {})
    model_dir = runtime_config.get('model_dir', './model_dir')
    
    tensorboard_config = callbacks_config.get('TensorBoard', {})
    if tensorboard_config.get('enabled', True):
        log_dir = tensorboard_config.get('log_dir', './logs/tensorboard')
        print(f"  1. Initial log_dir from config: {log_dir}")
        print(f"  2. Model dir: {model_dir}")
        
        # Current logic from enhanced_trainer.py
        if not os.path.isabs(log_dir) and not log_dir.startswith('./'):
            log_dir = os.path.join(model_dir, log_dir)
            print(f"  3. Path was modified to: {log_dir}")
        else:
            print(f"  3. Path kept as-is: {log_dir}")
        
        print(f"  ➤ Final callback log_dir: {log_dir}")
    print()
    
    # Check if paths match
    main_window_path = './logs/tensorboard'
    enhanced_trainer_path = './logs/tensorboard'
    
    print("🎯 Path Comparison:")
    print(f"  Main window will start TensorBoard with: {main_window_path}")
    print(f"  Enhanced trainer will save logs to: {enhanced_trainer_path}")
    
    if main_window_path == enhanced_trainer_path:
        print("  ✅ Paths match! This should work.")
    else:
        print("  ❌ Paths don't match! This is the problem.")
        print("     TensorBoard is looking in one place but logs are saved elsewhere.")
    print()
    
    # Test what happens if user changes the default
    print("🔧 Testing custom path scenario:")
    gui_cfg['callbacks']['TensorBoard']['log_dir'] = './custom/tensorboard'
    
    log_dir_main = './custom/tensorboard'  # main_window.py result
    log_dir_trainer = './custom/tensorboard'  # enhanced_trainer.py result
    
    print(f"  Main window: {log_dir_main}")
    print(f"  Enhanced trainer: {log_dir_trainer}")
    
    if log_dir_main == log_dir_trainer:
        print("  ✅ Custom paths also match!")
    else:
        print("  ❌ Custom paths don't match!")
    
    return True


if __name__ == '__main__':
    print("🚀 Debugging TensorBoard 'No dashboards are active' issue...\n")
    
    test_tensorboard_path_resolution()
    
    print("\n💡 Troubleshooting steps:")
    print("1. Check if the TensorBoard callback is actually being created during training")
    print("2. Verify that training logs are being written to the expected directory") 
    print("3. Ensure the TensorBoard server is pointed to the correct log directory")
    print("4. Look for any error messages during callback creation")
    
    print("\n🔍 To debug further:")
    print("- Check if './logs/tensorboard' directory exists after training starts")
    print("- Look for event files (*.tfevents.*) in the log directory")
    print("- Check the training logs for 'Added TensorBoard callback' message")
