"""
Enhanced custom preprocessing functions for ModelGardener.


All functions follow the nested wrapper pattern where:
- Outer function: Accepts configuration parameters (will be set in config.yaml)
- Inner wrapper function: Accepts (data, label) and returns (processed_data, processed_label)
- Configuration parameters are set at the outer function level

Example usage pattern:
def preprocessing_name(param1=default1, param2=default2):
    def wrapper(data, label):
        # Apply preprocessing logic here
        processed_data = apply_preprocessing(data, param1, param2)
        processed_label = process_label_if_needed(label)
        return processed_data, processed_label
    return wrapper

NOTE: With the new preprocessing pipeline, built-in preprocessing (sizing, normalization) 
is applied BEFORE custom preprocessing functions.
"""
def example_preprocessing_1(param1=1, param2=1):
    """
    Example custom preprocessing function
    and applies random horizontal flip for augmentation.
    
    Returns:
        Wrapper function that processes (data, label) tuples.
    """
    def wrapper(data, label):
        processed_data = data
        processed_label = label
        return processed_data, processed_label
    return wrapper

def example_preprocessing_2(param1=1, param2=1):
    """
    Example custom preprocessing function
    and applies random horizontal flip for augmentation.
    
    Returns:
        Wrapper function that processes (data, label) tuples.
    """
    def wrapper(data, label):
        processed_data = data
        processed_label = label
        return processed_data, processed_label
    return wrapper
