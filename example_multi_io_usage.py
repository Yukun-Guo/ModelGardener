"""
Example usage of the refactored ScalableDatasetLoader for multi-input/output models.

This example demonstrates how the refactored preprocessing and augmentation functions
now support both single tensor and tuple of tensors for multi-input/output models.
"""

import tensorflow as tf
from scalable_dataset_loader import ScalableDatasetLoader

def example_single_input_output():
    """Example with traditional single input/output model."""
    print("=== Single Input/Output Example ===")
    
    config = {
        'data': {
            'batch_size': 32,
            'preprocessing': {
                'Normalization': {
                    'enabled': True,
                    'method': 'zero-center'
                }
            },
            'augmentation': {
                'Horizontal Flip': {
                    'enabled': True,
                    'probability': 0.5
                }
            }
        }
    }
    
    # Create dataset with single image and label
    def single_data_generator():
        for i in range(5):
            image = tf.random.uniform([224, 224, 3], 0, 1, dtype=tf.float32)
            label = tf.constant(i % 3, dtype=tf.int32)
            yield image, label
    
    dataset = tf.data.Dataset.from_generator(
        single_data_generator,
        output_signature=(
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    
    loader = ScalableDatasetLoader(config)
    
    # Apply preprocessing and augmentation
    processed_dataset = loader.create_optimized_pipeline(dataset, 'train')
    
    print("Single input/output dataset created successfully!")
    for batch in processed_dataset.take(1):
        images, labels = batch
        print(f"Batch shape: images={images.shape}, labels={labels.shape}")

def example_multi_input_output():
    """Example with multi-input/output model (e.g., Siamese network or multi-task learning)."""
    print("\n=== Multi-Input/Output Example ===")
    
    config = {
        'data': {
            'batch_size': 32,
            'preprocessing': {
                'Normalization': {
                    'enabled': True,
                    'method': 'zero-center'
                },
                'Resizing': {
                    'enabled': True,
                    'target_size': {
                        'width': 224,
                        'height': 224
                    }
                }
            },
            'augmentation': {
                'Horizontal Flip': {
                    'enabled': True,
                    'probability': 0.3
                },
                'Brightness': {
                    'enabled': True,
                    'delta_range': 0.1,
                    'probability': 0.3
                }
            }
        }
    }
    
    # Create dataset with multiple inputs and outputs
    def multi_data_generator():
        for i in range(5):
            # Multi-input: two different sized images (e.g., for Siamese network)
            image1 = tf.random.uniform([224, 224, 3], 0, 1, dtype=tf.float32)
            image2 = tf.random.uniform([112, 112, 3], 0, 1, dtype=tf.float32)
            
            # Multi-output: multiple classification tasks
            label1 = tf.constant(i % 3, dtype=tf.int32)  # Task 1: 3 classes
            label2 = tf.constant(i % 2, dtype=tf.int32)  # Task 2: 2 classes
            
            yield (image1, image2), (label1, label2)
    
    dataset = tf.data.Dataset.from_generator(
        multi_data_generator,
        output_signature=(
            (tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
             tf.TensorSpec(shape=(112, 112, 3), dtype=tf.float32)),
            (tf.TensorSpec(shape=(), dtype=tf.int32),
             tf.TensorSpec(shape=(), dtype=tf.int32))
        )
    )
    
    loader = ScalableDatasetLoader(config)
    
    # Apply preprocessing and augmentation - now works with tuples!
    processed_dataset = loader.create_optimized_pipeline(dataset, 'train')
    
    print("Multi-input/output dataset created successfully!")
    for batch in processed_dataset.take(1):
        images, labels = batch
        if isinstance(images, tuple):
            print(f"Multi-input batch shapes: image1={images[0].shape}, image2={images[1].shape}")
        else:
            print(f"Single input batch shape: {images.shape}")
        
        if isinstance(labels, tuple):
            print(f"Multi-output batch shapes: label1={labels[0].shape}, label2={labels[1].shape}")
        else:
            print(f"Single output batch shape: {labels.shape}")

def example_mixed_scenarios():
    """Example showing that the same loader can handle both scenarios."""
    print("\n=== Mixed Scenarios Example ===")
    
    config = {
        'data': {
            'batch_size': 16,
            'preprocessing': {
                'Normalization': {
                    'enabled': True,
                    'method': 'standardize'
                }
            }
        }
    }
    
    loader = ScalableDatasetLoader(config)
    
    # Test with different input/output combinations
    test_cases = [
        # Single input, single output
        (tf.random.uniform([64, 64, 3]), tf.constant(0)),
        
        # Multi-input, single output
        ((tf.random.uniform([64, 64, 3]), tf.random.uniform([32, 32, 1])), tf.constant(1)),
        
        # Single input, multi-output
        (tf.random.uniform([128, 128, 3]), (tf.constant(0), tf.constant(1))),
        
        # Multi-input, multi-output
        ((tf.random.uniform([96, 96, 3]), tf.random.uniform([48, 48, 3])), 
         (tf.constant(2), tf.constant(0), tf.constant(1)))
    ]
    
    print("Testing various input/output combinations:")
    for i, (images, labels) in enumerate(test_cases):
        print(f"\nTest case {i+1}:")
        
        # Create a simple dataset
        dataset = tf.data.Dataset.from_tensors((images, labels))
        
        # Apply preprocessing
        processed_dataset = loader._apply_preprocessing(dataset, 'val')
        
        for processed_images, processed_labels in processed_dataset:
            if isinstance(processed_images, tuple):
                print(f"  Multi-input shapes: {[img.shape for img in processed_images]}")
            else:
                print(f"  Single input shape: {processed_images.shape}")
            
            if isinstance(processed_labels, tuple):
                print(f"  Multi-output: {len(processed_labels)} outputs")
            else:
                print(f"  Single output")

if __name__ == "__main__":
    # Run all examples
    example_single_input_output()
    example_multi_input_output()
    example_mixed_scenarios()
    
    print("\n" + "="*50)
    print("All examples completed successfully!")
    print("The ScalableDatasetLoader now supports:")
    print("✓ Single input/output models (traditional)")
    print("✓ Multi-input models (e.g., Siamese networks)")
    print("✓ Multi-output models (e.g., multi-task learning)")
    print("✓ Multi-input/multi-output models")
    print("="*50)
