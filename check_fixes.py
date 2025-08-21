#!/usr/bin/env python3
"""
Simple test to check if the fix methods were added to all custom function groups.
"""

import sys
import os

def check_methods_added():
    """Check if the required methods were added to all custom function groups."""
    
    # Check each group file for the required methods
    groups_to_check = {
        'data_loader_group.py': ['set_data_loader_config', 'load_custom_data_loader_from_metadata'],
        'optimizer_group.py': ['set_optimizer_config', 'load_custom_optimizer_from_metadata'],
        'loss_functions_group.py': ['set_loss_config', 'load_custom_loss_from_metadata'], 
        'metrics_group.py': ['set_metrics_config', 'load_custom_metric_from_metadata'],
        'callbacks_group.py': ['set_callbacks_config', 'load_custom_callback_from_metadata'],
        'preprocessing_group.py': ['set_preprocessing_config', 'load_custom_preprocessing_from_metadata']
    }
    
    results = {}
    
    for filename, required_methods in groups_to_check.items():
        print(f"📁 Checking {filename}...")
        
        if not os.path.exists(filename):
            print(f"❌ File {filename} not found")
            results[filename] = False
            continue
            
        with open(filename, 'r') as f:
            content = f.read()
        
        file_results = {}
        for method in required_methods:
            if f'def {method}(' in content:
                print(f"  ✅ {method} - FOUND")
                file_results[method] = True
            else:
                print(f"  ❌ {method} - MISSING")
                file_results[method] = False
        
        # Check if all methods are present
        all_present = all(file_results.values())
        results[filename] = all_present
        
        status = "✅ COMPLETE" if all_present else "❌ INCOMPLETE"
        print(f"  Status: {status}")
        print()
    
    # Overall results
    total_groups = len(results)
    successful_groups = sum(results.values())
    
    print(f"🎯 Overall Results: {successful_groups}/{total_groups} groups have all required methods")
    
    for filename, complete in results.items():
        group_name = filename.replace('_group.py', '').replace('_', ' ').title()
        status = "✅ READY" if complete else "❌ NEEDS WORK"
        print(f"  {group_name}: {status}")
    
    if successful_groups == total_groups:
        print("\n🎉 ALL GROUPS HAVE BEEN FIXED! ✅")
        print("All custom function groups now have the required configuration loading methods.")
        return True
    else:
        print(f"\n❌ {total_groups - successful_groups} groups still need fixes")
        return False

def check_main_window_updates():
    """Check if main_window.py was updated to handle all groups."""
    print("📁 Checking main_window.py updates...")
    
    if not os.path.exists('main_window.py'):
        print("❌ main_window.py not found")
        return False
    
    with open('main_window.py', 'r') as f:
        content = f.read()
    
    # Check for the updated _apply_config_to_custom_groups method
    checks = {
        'set_data_loader_config': 'data_loader_group.set_data_loader_config' in content,
        'set_optimizer_config': 'optimizer_group.set_optimizer_config' in content,
        'set_loss_config': 'loss_functions_group.set_loss_config' in content,
        'set_metrics_config': 'metrics_group.set_metrics_config' in content,
        'set_callbacks_config': 'callbacks_group.set_callbacks_config' in content,
        'set_preprocessing_config': 'preprocessing_group.set_preprocessing_config' in content,
        'metadata loading optimizers': 'load_custom_optimizer_from_metadata' in content,
        'metadata loading metrics': 'load_custom_metric_from_metadata' in content
    }
    
    for check_name, check_result in checks.items():
        status = "✅ FOUND" if check_result else "❌ MISSING"
        print(f"  {check_name}: {status}")
    
    all_checks_passed = all(checks.values())
    
    status = "✅ UPDATED" if all_checks_passed else "❌ NEEDS UPDATES"
    print(f"  Status: {status}")
    
    return all_checks_passed

def main():
    """Main check function."""
    print("🔍 Checking if all custom function groups have been fixed...")
    print("=" * 60)
    
    groups_ok = check_methods_added()
    print()
    main_window_ok = check_main_window_updates()
    
    print("\n" + "=" * 60)
    
    if groups_ok and main_window_ok:
        print("🎉 COMPREHENSIVE FIX COMPLETE! ✅")
        print("\nAll custom function groups now have:")
        print("  • Configuration loading methods (set_*_config)")
        print("  • Metadata-based custom function loading") 
        print("  • Integration with main_window.py")
        print("\nThe bug fix has been applied to all custom function groups!")
        return True
    else:
        print("❌ FIXES INCOMPLETE")
        if not groups_ok:
            print("  • Some groups are missing required methods")
        if not main_window_ok:
            print("  • main_window.py needs updates")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
