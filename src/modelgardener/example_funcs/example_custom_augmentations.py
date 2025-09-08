"""
Custom augmentation functions for ModelGardener.

All functions follow the nested wrapper pattern where:
- Outer function: Accepts configuration parameters (will be set in config.yaml)
- Inner wrapper function: Accepts (data, label) and returns (modified_data, modified_label)
- Configuration parameters are set at the outer function level

Example usage pattern:
def augmentation_name(param1=default1, param2=default2):
    def wrapper(data, label):
        # Apply augmentation logic here
        modified_data = apply_augmentation(data, param1, param2)
        modified_label = modify_label_if_needed(label, augmentation_params)
        return modified_data, modified_label
    return wrapper
"""


def example_augmentation_1(param1=1,param2=2):
    def wrapper(data, label):
        # Apply augmentation logic here
        modified_data = data  # Replace with actual augmentation logic
        modified_label = label  # Replace with actual label modification logic if needed
        return modified_data, modified_label
    return wrapper

def example_augmentation_2(param1=1,param2=2):
    def wrapper(data, label):
        # Apply augmentation logic here
        modified_data = data  # Replace with actual augmentation logic
        modified_label = label  # Replace with actual label modification logic if needed
        return modified_data, modified_label
    return wrapper