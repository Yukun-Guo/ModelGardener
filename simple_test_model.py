#!/usr/bin/env python3
"""
Simple test model without TensorFlow dependencies for testing custom model loading.
"""

def create_simple_model(input_shape=(224, 224, 3), num_classes=1000):
    """Simple test model function for demonstrating custom model loading."""
    print(f"Creating simple model with input_shape={input_shape}, num_classes={num_classes}")
    # Return a mock model object
    return {"type": "test_model", "input_shape": input_shape, "num_classes": num_classes}

class SimpleTestModel:
    """Test custom model class without TensorFlow dependencies."""
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=1000):
        self.input_shape = input_shape
        self.num_classes = num_classes
        print(f"SimpleTestModel initialized with input_shape={input_shape}, num_classes={num_classes}")
    
    def forward(self, x):
        """Mock forward pass."""
        return x

def helper_function():
    """This is just a helper function, not a model."""
    return "helper"

def not_a_model():
    """This function doesn't look like a model."""
    return 42

if __name__ == "__main__":
    print("Simple test model file loaded successfully!")
    model = create_simple_model()
    print(f"Model created: {model}")
    
    test_model = SimpleTestModel()
    print(f"Test model class instantiated: {test_model}")
