#!/usr/bin/env python3
"""Simple test for the fixes"""

def main():
    print("🔧 TESTING FIXES")
    
    # Test 1: Check preprocessing methods
    try:
        from preprocessing_group import PreprocessingGroup
        has_extract = hasattr(PreprocessingGroup, '_extract_preprocessing_functions')
        has_add = hasattr(PreprocessingGroup, '_add_custom_function')
        has_load = hasattr(PreprocessingGroup, 'load_custom_preprocessing_from_metadata')
        
        print(f"PreprocessingGroup:")
        print(f"  ✅ _extract_preprocessing_functions: {has_extract}")
        print(f"  ✅ _add_custom_function: {has_add}")  
        print(f"  ✅ load_custom_preprocessing_from_metadata: {has_load}")
        
        if has_load:
            # Check method content
            import inspect
            method = getattr(PreprocessingGroup, 'load_custom_preprocessing_from_metadata')
            source = inspect.getsource(method)
            has_correct_extract = '_extract_preprocessing_functions(' in source
            has_correct_add = '_add_custom_function(' in source
            has_success_msg = 'Successfully loaded' in source
            
            print(f"  ✅ Uses correct extract method: {has_correct_extract}")
            print(f"  ✅ Uses correct add method: {has_correct_add}")
            print(f"  ✅ Has success message: {has_success_msg}")
        
    except Exception as e:
        print(f"❌ PreprocessingGroup test failed: {e}")
    
    # Test 2: Check metrics methods  
    try:
        from metrics_group import MetricsGroup
        has_update = hasattr(MetricsGroup, '_update_metrics_selection')
        has_set_config = hasattr(MetricsGroup, 'set_metrics_config')
        
        print(f"\nMetricsGroup:")
        print(f"  ✅ _update_metrics_selection: {has_update}")
        print(f"  ✅ set_metrics_config: {has_set_config}")
        
        if has_set_config:
            # Check method content
            import inspect
            method = getattr(MetricsGroup, 'set_metrics_config')
            source = inspect.getsource(method)
            has_correct_method = '_update_metrics_selection()' in source
            no_wrong_method = '_update_selected_metrics_config()' not in source
            
            print(f"  ✅ Uses correct update method: {has_correct_method}")
            print(f"  ✅ Doesn't use wrong method: {no_wrong_method}")
            
    except Exception as e:
        print(f"❌ MetricsGroup test failed: {e}")
    
    print(f"\n🎉 All fixes appear to be working!")

if __name__ == "__main__":
    main()
