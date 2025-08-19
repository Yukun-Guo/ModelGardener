#!/usr/bin/env python3
"""
Simple test script to verify parameter tree list initialization logic
without requiring heavy dependencies.
"""

def get_parameter_tooltip(param_name):
    """Simplified version of get_parameter_tooltip for testing."""
    return f"Tooltip for {param_name}"

def create_test_config():
    """Create a simple test configuration."""
    return {
        'model': {
            'backbone_type': 'resnet',
            'activation': 'relu',
        },
        'runtime': {
            'mixed_precision': None,
            'distribution_strategy': 'mirrored'
        }
    }

def dict_to_params_test(data, name="Config"):
    """Simplified version of dict_to_params for testing list parameters."""
    if isinstance(data, dict):
        children = []
        for key, value in data.items():
            if isinstance(value, dict):
                # Nested dictionary - create a group
                group_param = dict_to_params_test(value, key)
                group_param['tip'] = get_parameter_tooltip(key)
                children.append(group_param)
            else:
                # Handle special choice parameters
                if key == 'backbone_type':
                    values = ['resnet', 'efficientnet', 'mobilenet', 'vit', 'densenet']
                    # Ensure current value is valid, default to first item if not
                    current_value = value if value in values else values[0]
                    children.append({
                        'name': key,
                        'type': 'list',
                        'values': values,
                        'value': current_value,
                        'tip': get_parameter_tooltip(key)
                    })
                elif key == 'activation':
                    values = ['relu', 'swish', 'gelu', 'leaky_relu', 'tanh']
                    current_value = value if value in values else values[0]
                    children.append({
                        'name': key,
                        'type': 'list',
                        'values': values,
                        'value': current_value,
                        'tip': get_parameter_tooltip(key)
                    })
                elif key == 'mixed_precision':
                    values = ['None', 'float16', 'bfloat16']  # Convert None to 'None' string for dropdown
                    # Handle None value properly
                    if value is None:
                        current_value = 'None'
                    else:
                        current_value = value if value in values else values[0]
                    children.append({
                        'name': key,
                        'type': 'list',
                        'values': values,
                        'value': current_value,
                        'tip': get_parameter_tooltip(key)
                    })
                elif key == 'distribution_strategy':
                    values = ['mirrored', 'multi_worker_mirrored', 'tpu', 'parameter_server']
                    current_value = value if value in values else values[0]
                    children.append({
                        'name': key,
                        'type': 'list',
                        'values': values,
                        'value': current_value,
                        'tip': get_parameter_tooltip(key)
                    })
                else:
                    # Default parameter handling
                    if isinstance(value, bool):
                        children.append({
                            'name': key,
                            'type': 'bool',
                            'value': value,
                            'tip': get_parameter_tooltip(key)
                        })
                    else:
                        children.append({
                            'name': key,
                            'type': 'str',
                            'value': str(value) if value is not None else '',
                            'tip': get_parameter_tooltip(key)
                        })
        
        return {
            'name': name,
            'type': 'group',
            'children': children,
            'tip': get_parameter_tooltip(name)
        }
    else:
        return {
            'name': name,
            'type': 'str',
            'value': str(data) if data is not None else '',
            'tip': get_parameter_tooltip(name)
        }

def test_list_parameter_initialization():
    """Test that list parameters are initialized correctly."""
    
    print("Testing list parameter initialization...")
    
    # Create test config
    config = create_test_config()
    print(f"Test config: {config}")
    
    # Convert to parameter structure
    param_dict = dict_to_params_test(config, "Configuration")
    
    def find_param_by_name(param_dict, name):
        """Recursively find a parameter by name."""
        if isinstance(param_dict, dict):
            if param_dict.get('name') == name:
                return param_dict
            if 'children' in param_dict:
                for child in param_dict['children']:
                    result = find_param_by_name(child, name)
                    if result:
                        return result
        return None
    
    # Test parameters
    test_cases = [
        ('backbone_type', 'resnet'),
        ('activation', 'relu'),
        ('mixed_precision', 'None'),  # Should be converted from None to 'None'
        ('distribution_strategy', 'mirrored')
    ]
    
    all_passed = True
    
    for param_name, expected_value in test_cases:
        param = find_param_by_name(param_dict, param_name)
        if param:
            actual_value = param.get('value')
            values_list = param.get('values', [])
            
            print(f"\n{param_name}:")
            print(f"  - Values: {values_list}")
            print(f"  - Current value: {actual_value}")
            print(f"  - Expected value: {expected_value}")
            
            if actual_value == expected_value:
                print(f"  ✓ Value matches expected")
            else:
                print(f"  ✗ Value doesn't match expected")
                all_passed = False
                
            if actual_value in values_list:
                print(f"  ✓ Current value is in values list")
            else:
                print(f"  ✗ Current value is not in values list")
                all_passed = False
        else:
            print(f"✗ Could not find parameter: {param_name}")
            all_passed = False
    
    if all_passed:
        print(f"\n✓ All tests passed!")
    else:
        print(f"\n✗ Some tests failed!")
    
    return all_passed

if __name__ == "__main__":
    test_list_parameter_initialization()
