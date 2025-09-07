"""
Enhanced custom data loader functions for ModelGardener

Supports:
- Multi-input and multi-output models
- 2D and 3D data
- Multiple task types (classification, segmentation, object detection)

This file demonstrates how to create custom data loader functions and classes that can be
dynamically loaded into the ModelGardener application. Data loaders can be either:

1. Functions that return a tf.data.Dataset
2. Classes that provide data loading functionality

For functions:
- Should return a tf.data.Dataset instance
- Can accept configuration parameters like batch_size, shuffle, etc.
- Should handle both training and validation data paths

For classes:
- Should provide methods to create datasets
- Can have __init__ method with configuration parameters
- Should implement data loading logic
"""

import os
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Dict, Union, Any
from sklearn.model_selection import train_test_split
from .utils import TaskType, DataDimension, detect_data_dimension, infer_task_type

def enhanced_image_data_loader(data_dir: str,
                             batch_size: int = 32,
                             image_size: List[int] = [224, 224],
                             data_dimension: str = '2d',
                             task_type: str = 'classification',
                             shuffle: bool = True,
                             buffer_size: int = 10000,
                             augment: bool = False,
                             label_dir: str = None,
                             multi_input: bool = False,
                             input_dirs: List[str] = None) -> tf.data.Dataset:
    """
    Enhanced image data loader with support for 2D/3D data, multi-inputs, and different tasks.
    
    Args:
        data_dir: Path to directory containing image files
        batch_size: Batch size for the dataset
        image_size: Target image size [height, width] or [height, width, depth] for 3D
        data_dimension: '2d' or '3d'
        task_type: 'classification', 'segmentation', 'object_detection'
        shuffle: Whether to shuffle the dataset
        buffer_size: Buffer size for shuffling
        augment: Whether to apply data augmentation
        label_dir: Path to labels (for segmentation/detection tasks)
        multi_input: Whether to create multi-input dataset
        input_dirs: List of input directories for multi-input
    
    Returns:
        tf.data.Dataset: Dataset ready for training/validation
    """
    # Determine file extensions based on data dimension
    if data_dimension == '3d':
        # Common 3D medical imaging formats
        image_extensions = ['.nii', '.nii.gz', '.mha', '.mhd', '.nrrd', '.dcm']
    else:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    if multi_input and input_dirs:
        # Multi-input case
        all_input_files = []
        for input_dir in input_dirs:
            input_files = get_image_files(input_dir, image_extensions)
            all_input_files.append(input_files)
        
        # Ensure all input lists have the same length
        min_length = min(len(files) for files in all_input_files)
        all_input_files = [files[:min_length] for files in all_input_files]
        
        # Create dataset from multi-input file paths
        input_datasets = []
        for input_files in all_input_files:
            ds = tf.data.Dataset.from_tensor_slices(input_files)
            input_datasets.append(ds)
        
        # Zip datasets together
        dataset = tf.data.Dataset.zip(tuple(input_datasets))
        
        # Load and preprocess multi-input data
        def load_and_preprocess_multi_input(file_paths):
            processed_inputs = []
            for file_path in file_paths:
                if data_dimension == '3d':
                    image = load_3d_image(file_path, image_size)
                else:
                    image = load_2d_image(file_path, image_size)
                processed_inputs.append(image)
            return tuple(processed_inputs)
        
        dataset = dataset.map(load_and_preprocess_multi_input, 
                             num_parallel_calls=tf.data.AUTOTUNE)
    else:
        # Single input case
        image_files = get_image_files(data_dir, image_extensions)
        if not image_files:
            raise ValueError(f"No image files found in {data_dir}")
        
        dataset = tf.data.Dataset.from_tensor_slices(image_files)
        
        # Load and preprocess images
        def load_and_preprocess_image(path):
            if data_dimension == '3d':
                return load_3d_image(path, image_size)
            else:
                return load_2d_image(path, image_size)
        
        dataset = dataset.map(load_and_preprocess_image, 
                             num_parallel_calls=tf.data.AUTOTUNE)
    
    # Handle labels based on task type
    if task_type == 'segmentation' and label_dir:
        # Load segmentation masks
        label_files = get_image_files(label_dir, image_extensions)
        label_dataset = tf.data.Dataset.from_tensor_slices(label_files)
        
        def load_label_mask(path):
            if data_dimension == '3d':
                return load_3d_image(path, image_size, is_label=True)
            else:
                return load_2d_image(path, image_size, is_label=True)
        
        label_dataset = label_dataset.map(load_label_mask, 
                                        num_parallel_calls=tf.data.AUTOTUNE)
        
        # Combine data and labels
        dataset = tf.data.Dataset.zip((dataset, label_dataset))
    
    elif task_type == 'classification':
        # Extract labels from directory structure or filename
        if multi_input:
            # For multi-input, use the first input for label extraction
            label_source = all_input_files[0] if input_dirs else image_files
        else:
            label_source = image_files
            
        labels = extract_classification_labels(label_source)
        label_dataset = tf.data.Dataset.from_tensor_slices(labels)
        
        # Combine data and labels
        dataset = tf.data.Dataset.zip((dataset, label_dataset))
    
    elif task_type == 'object_detection':
        # Load bounding boxes and class labels
        # This would require annotation files (COCO, YOLO format, etc.)
        # Simplified implementation
        dummy_labels = tf.zeros((4,))  # Placeholder for bounding boxes
        label_dataset = tf.data.Dataset.from_tensor_slices([dummy_labels] * len(image_files))
        dataset = tf.data.Dataset.zip((dataset, label_dataset))
    
    # Apply augmentation if requested
    if augment:
        dataset = dataset.map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Shuffle and batch
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def get_image_files(directory: str, extensions: List[str]) -> List[str]:
    """Get list of image files with specified extensions."""
    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                image_files.append(os.path.join(root, file))
    return sorted(image_files)

def load_2d_image(file_path: tf.Tensor, target_size: List[int], is_label: bool = False) -> tf.Tensor:
    """Load and preprocess 2D image."""
    image = tf.io.read_file(file_path)
    image = tf.image.decode_image(image, channels=3 if not is_label else 1, expand_animations=False)
    image = tf.image.resize(image, target_size[:2])
    
    if is_label:
        # Keep labels as integers for segmentation
        image = tf.cast(image, tf.int32)
    else:
        # Normalize to [0, 1] for images
        image = tf.cast(image, tf.float32) / 255.0
    
    return image

def load_3d_image(file_path: tf.Tensor, target_size: List[int], is_label: bool = False) -> tf.Tensor:
    """Load and preprocess 3D image (medical imaging, etc.)."""
    # This is a placeholder implementation
    # In practice, you would use libraries like nibabel, SimpleITK, etc.
    # For now, we'll simulate 3D data by creating a synthetic volume
    
    if len(target_size) == 2:
        target_size = target_size + [32]  # Default depth
    
    # Simulate loading 3D data
    if is_label:
        volume = tf.random.uniform(target_size + [1], 0, 2, dtype=tf.int32)
    else:
        volume = tf.random.normal(target_size + [1], dtype=tf.float32)
        volume = tf.clip_by_value(volume, 0.0, 1.0)
    
    return volume

def extract_classification_labels(file_paths: List[str]) -> List[int]:
    """Extract classification labels from file paths or directory structure."""
    # Extract class names from directory structure
    class_names = []
    for file_path in file_paths:
        # Assuming directory structure like: data_dir/class_name/image.jpg
        class_name = os.path.basename(os.path.dirname(file_path))
        class_names.append(class_name)
    
    # Convert to numerical labels
    unique_classes = sorted(list(set(class_names)))
    class_to_label = {cls: idx for idx, cls in enumerate(unique_classes)}
    labels = [class_to_label[cls] for cls in class_names]
    
    return labels

def apply_augmentation(data, label):
    """Apply simple augmentation to data."""
    if isinstance(data, tuple):
        # Multi-input case
        augmented_data = []
        for input_data in data:
            # Apply basic augmentation
            augmented = tf.image.random_flip_left_right(input_data)
            augmented = tf.image.random_brightness(augmented, max_delta=0.1)
            augmented_data.append(augmented)
        return tuple(augmented_data), label
    else:
        # Single input case
        augmented = tf.image.random_flip_left_right(data)
        augmented = tf.image.random_brightness(augmented, max_delta=0.1)
        return augmented, label
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def Custom_load_cifar10_npz_data(train_dir: str = "./data", 
                                 val_dir: str = "./data",
                                 npz_file_path: str = "./data/cifar10.npz",
                                 batch_size: int = 32,
                                 shuffle: bool = True,
                                 buffer_size: int = 1000,
                                 validation_split: float = 0.2,
                                 **kwargs):
    """
    Custom CIFAR-10 NPZ data loader for ModelGardener.
    
    This function loads CIFAR-10 data from an NPZ file and returns
    training and validation datasets.
    
    Args:
        train_dir: Directory path (used for compatibility)
        val_dir: Directory path (used for compatibility) 
        npz_file_path: Path to the NPZ file containing CIFAR-10 data
        batch_size: Batch size for datasets
        shuffle: Whether to shuffle the data
        buffer_size: Buffer size for shuffling
        validation_split: Fraction of data to use for validation
        **kwargs: Additional parameters (ignored)
        
    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset]: Training and validation datasets
    """
    print(f"🔍 Loading CIFAR-10 data from: {npz_file_path}")
    
    # Load NPZ file
    if not os.path.exists(npz_file_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_file_path}")
    
    data = np.load(npz_file_path)
    images = data['x'].astype(np.float32) / 255.0  # Normalize to [0, 1]
    labels = data['y'].astype(np.int32)
    
    print(f"📊 Loaded {len(images)} images with shape {images.shape[1:]}")
    print(f"🎯 Found {len(np.unique(labels))} unique classes")
    
    # Split into train and validation
    train_indices = int(len(images) * (1 - validation_split))
    
    train_images = images[:train_indices]
    train_labels = labels[:train_indices]
    val_images = images[train_indices:]
    val_labels = labels[train_indices:]
    
    # Convert labels to categorical (one-hot encoding)
    num_classes = len(np.unique(labels))
    train_labels_categorical = tf.keras.utils.to_categorical(train_labels, num_classes)
    val_labels_categorical = tf.keras.utils.to_categorical(val_labels, num_classes)
    
    print(f"🚂 Training set: {len(train_images)} samples")
    print(f"✅ Validation set: {len(val_images)} samples")
    
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels_categorical))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels_categorical))
    
    # Apply shuffling if requested
    if shuffle:
        train_dataset = train_dataset.shuffle(buffer_size=buffer_size)
    
    # Batch the datasets
    train_dataset = train_dataset.batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)
    
    # Prefetch for performance
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset


def load_cifar10_npz_data(data_dir: str,
                          npz_file: str = "cifar10.npz", 
                          batch_size: int = 32,
                          validation_split: float = 0.2,
                          shuffle: bool = True,
                          buffer_size: int = 10000,
                          split: str = 'train',
                          seed: int = 42,
                          **kwargs) -> tf.data.Dataset:
    """
    Load CIFAR-10 subset data from NPZ file for ModelGardener.
    
    This data loader is specifically designed for the CIFAR-10 subset dataset
    generated by test_generate_subset.py. It handles train/validation splits,
    normalization, and categorical encoding automatically.
    
    Args:
        data_dir: Directory containing the NPZ file
        npz_file: Name of the NPZ file (default: "cifar10.npz")
        batch_size: Batch size for the dataset
        validation_split: Fraction of data to use for validation (0.0-1.0)
        shuffle: Whether to shuffle the dataset
        buffer_size: Buffer size for shuffling
        split: Which split to return ('train' or 'val')
        seed: Random seed for reproducibility
    
    Returns:
        tf.data.Dataset: Dataset ready for training/validation
        
    Example:
        # Load training data
        train_ds = load_cifar10_npz_data(
            data_dir="./example_data",
            batch_size=32,
            split='train'
        )
        
        # Load validation data
        val_ds = load_cifar10_npz_data(
            data_dir="./example_data", 
            batch_size=32,
            split='val'
        )
    """
    # Construct full path to NPZ file
    npz_path = os.path.join(data_dir, npz_file)
    
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    
    # Load data from NPZ file
    data = np.load(npz_path)
    x_data = data['x']  # Shape: (1000, 32, 32, 3)
    y_data = data['y']  # Shape: (1000,)
    
    print(f"Loaded CIFAR-10 data: {x_data.shape}, labels: {y_data.shape}")
    print(f"Classes: {np.unique(y_data)}")
    
    # Normalize images to [0, 1]
    x_data = x_data.astype(np.float32) / 255.0
    
    # Convert labels to categorical (one-hot encoding)
    num_classes = len(np.unique(y_data))
    y_data_categorical = tf.keras.utils.to_categorical(y_data, num_classes)
    
    # Split data into train and validation
    if validation_split > 0.0:
        x_train, x_val, y_train, y_val = train_test_split(
            x_data, y_data_categorical, 
            test_size=validation_split, 
            random_state=seed,
            stratify=y_data  # Ensure balanced split
        )
    else:
        x_train, y_train = x_data, y_data_categorical
        x_val, y_val = x_data, y_data_categorical  # Use same data if no split
    
    # Select the appropriate split
    if split == 'train':
        x_split, y_split = x_train, y_train
        print(f"Using training data: {x_split.shape[0]} samples")
    elif split == 'val' or split == 'validation':
        x_split, y_split = x_val, y_val
        print(f"Using validation data: {x_split.shape[0]} samples")
    else:
        raise ValueError(f"Unknown split: {split}. Use 'train' or 'val'")
    
    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((x_split, y_split))
    
    # Shuffle if requested (only for training)
    if shuffle and split == 'train':
        dataset = dataset.shuffle(buffer_size, seed=seed)
    
    # Batch the data
    dataset = dataset.batch(batch_size)
    
    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


class CIFAR10NPZDataLoader:
    """
    Class-based CIFAR-10 NPZ data loader for ModelGardener.
    
    This class provides a convenient interface for loading CIFAR-10 subset data
    with configurable parameters and methods for accessing train/val splits.
    """
    
    def __init__(self, 
                 data_dir: str,
                 npz_file: str = "cifar10.npz",
                 batch_size: int = 32,
                 validation_split: float = 0.2,
                 shuffle: bool = True,
                 buffer_size: int = 10000,
                 seed: int = 42):
        """
        Initialize the CIFAR-10 NPZ data loader.
        
        Args:
            data_dir: Directory containing the NPZ file
            npz_file: Name of the NPZ file
            batch_size: Batch size for the dataset
            validation_split: Fraction of data for validation
            shuffle: Whether to shuffle the training data
            buffer_size: Buffer size for shuffling
            seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.npz_file = npz_file
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.seed = seed
        
        # Load and preprocess data once
        self._load_data()
    
    def _load_data(self):
        """Load and preprocess data from NPZ file."""
        npz_path = os.path.join(self.data_dir, self.npz_file)
        
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")
        
        data = np.load(npz_path)
        self.x_data = data['x'].astype(np.float32) / 255.0
        self.y_data = data['y']
        
        # Convert to categorical
        self.num_classes = len(np.unique(self.y_data))
        self.y_data_categorical = tf.keras.utils.to_categorical(self.y_data, self.num_classes)
        
        # Split data
        if self.validation_split > 0.0:
            self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
                self.x_data, self.y_data_categorical,
                test_size=self.validation_split,
                random_state=self.seed,
                stratify=self.y_data
            )
        else:
            self.x_train = self.x_val = self.x_data
            self.y_train = self.y_val = self.y_data_categorical
    
    def get_dataset(self, split: str = 'train') -> tf.data.Dataset:
        """
        Get dataset for specified split.
        
        Args:
            split: Which split to return ('train' or 'val')
            
        Returns:
            tf.data.Dataset: The requested dataset
        """
        if split == 'train':
            return self._create_dataset(self.x_train, self.y_train, shuffle=self.shuffle)
        elif split in ['val', 'validation']:
            return self._create_dataset(self.x_val, self.y_val, shuffle=False)
        else:
            raise ValueError(f"Unknown split: {split}. Use 'train' or 'val'")
    
    def _create_dataset(self, x_data: np.ndarray, y_data: np.ndarray, shuffle: bool = True) -> tf.data.Dataset:
        """Create TensorFlow dataset from numpy arrays."""
        dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
        
        if shuffle:
            dataset = dataset.shuffle(self.buffer_size, seed=self.seed)
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    @property
    def input_shape(self) -> Tuple[int, int, int]:
        """Get input shape of the data."""
        return self.x_data.shape[1:]
    
    @property
    def num_samples_train(self) -> int:
        """Get number of training samples."""
        return len(self.x_train)
    
    @property
    def num_samples_val(self) -> int:
        """Get number of validation samples."""
        return len(self.x_val)


# Alternative simple function for basic usage
def simple_cifar10_loader(data_dir: str,
                         batch_size: int = 32,
                         split: str = 'train',
                         **kwargs) -> tf.data.Dataset:
    """
    Simple CIFAR-10 NPZ loader with minimal configuration.
    
    Args:
        data_dir: Directory containing cifar10.npz file
        batch_size: Batch size for the dataset
        split: Which split to return ('train' or 'val')
        **kwargs: Additional arguments (for compatibility)
        
    Returns:
        tf.data.Dataset: Dataset ready for training/validation
    """
    return load_cifar10_npz_data(
        data_dir=data_dir,
        npz_file="cifar10.npz",
        batch_size=batch_size,
        split=split,
        validation_split=0.2,
        shuffle=True
    )

def multi_input_data_loader(input_data_dirs: List[str],
                          batch_size: int = 32,
                          image_sizes: List[List[int]] = None,
                          task_type: str = 'classification',
                          label_dir: str = None,
                          **kwargs) -> tf.data.Dataset:
    """
    Data loader for multi-input models.
    
    Args:
        input_data_dirs: List of directories containing different input modalities
        batch_size: Batch size for the dataset
        image_sizes: List of target sizes for each input modality
        task_type: Type of task ('classification', 'segmentation')
        label_dir: Directory containing labels
        **kwargs: Additional arguments
        
    Returns:
        tf.data.Dataset: Multi-input dataset
    """
    if image_sizes is None:
        image_sizes = [[224, 224]] * len(input_data_dirs)
    
    return enhanced_image_data_loader(
        data_dir=input_data_dirs[0],  # Primary input directory
        batch_size=batch_size,
        image_size=image_sizes[0],
        task_type=task_type,
        label_dir=label_dir,
        multi_input=True,
        input_dirs=input_data_dirs,
        **kwargs
    )

def volumetric_data_loader(data_dir: str,
                         batch_size: int = 4,  # Smaller batch for 3D data
                         volume_size: List[int] = [64, 64, 64],
                         task_type: str = 'classification',
                         label_dir: str = None,
                         file_format: str = 'nifti',
                         **kwargs) -> tf.data.Dataset:
    """
    Data loader for 3D volumetric data (medical imaging, etc.).
    
    Args:
        data_dir: Directory containing volumetric data
        batch_size: Batch size (typically smaller for 3D data)
        volume_size: Target volume size [height, width, depth]
        task_type: Type of task ('classification', 'segmentation')
        label_dir: Directory containing label volumes
        file_format: File format ('nifti', 'dicom', etc.)
        **kwargs: Additional arguments
        
    Returns:
        tf.data.Dataset: 3D volumetric dataset
    """
    return enhanced_image_data_loader(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=volume_size,
        data_dimension='3d',
        task_type=task_type,
        label_dir=label_dir,
        **kwargs
    )

def segmentation_data_loader(image_dir: str,
                           mask_dir: str,
                           batch_size: int = 16,
                           image_size: List[int] = [256, 256],
                           num_classes: int = 2,
                           data_dimension: str = '2d',
                           **kwargs) -> tf.data.Dataset:
    """
    Specialized data loader for segmentation tasks.
    
    Args:
        image_dir: Directory containing input images
        mask_dir: Directory containing segmentation masks
        batch_size: Batch size for the dataset
        image_size: Target image size
        num_classes: Number of segmentation classes
        data_dimension: '2d' or '3d'
        **kwargs: Additional arguments
        
    Returns:
        tf.data.Dataset: Segmentation dataset with images and masks
    """
    return enhanced_image_data_loader(
        data_dir=image_dir,
        batch_size=batch_size,
        image_size=image_size,
        data_dimension=data_dimension,
        task_type='segmentation',
        label_dir=mask_dir,
        **kwargs
    )

def object_detection_data_loader(image_dir: str,
                               annotation_file: str,
                               batch_size: int = 8,
                               image_size: List[int] = [416, 416],
                               max_objects: int = 100,
                               **kwargs) -> tf.data.Dataset:
    """
    Data loader for object detection tasks.
    
    Args:
        image_dir: Directory containing images
        annotation_file: Path to annotations file (COCO format, etc.)
        batch_size: Batch size for the dataset
        image_size: Target image size
        max_objects: Maximum number of objects per image
        **kwargs: Additional arguments
        
    Returns:
        tf.data.Dataset: Object detection dataset
    """
    # This is a simplified implementation
    # In practice, you would parse COCO annotations, YOLO format, etc.
    return enhanced_image_data_loader(
        data_dir=image_dir,
        batch_size=batch_size,
        image_size=image_size,
        task_type='object_detection',
        **kwargs
    )

class AdaptiveDataLoader:
    """
    Adaptive data loader class that can handle different data types and tasks.
    """
    
    def __init__(self, 
                 data_config: Dict[str, Any],
                 task_type: str = 'classification',
                 data_dimension: str = '2d'):
        """
        Initialize adaptive data loader.
        
        Args:
            data_config: Configuration dictionary for data loading
            task_type: Type of task
            data_dimension: Data dimension
        """
        self.data_config = data_config
        self.task_type = task_type
        self.data_dimension = data_dimension
        
    def create_dataset(self, split: str = 'train') -> tf.data.Dataset:
        """
        Create dataset based on configuration.
        
        Args:
            split: Dataset split ('train', 'val', 'test')
            
        Returns:
            tf.data.Dataset: Configured dataset
        """
        config = self.data_config.get(split, self.data_config)
        
        if self.task_type == 'segmentation':
            return segmentation_data_loader(
                image_dir=config['image_dir'],
                mask_dir=config['mask_dir'],
                data_dimension=self.data_dimension,
                **config.get('loader_params', {})
            )
        elif self.task_type == 'object_detection':
            return object_detection_data_loader(
                image_dir=config['image_dir'],
                annotation_file=config['annotation_file'],
                **config.get('loader_params', {})
            )
        elif config.get('multi_input', False):
            return multi_input_data_loader(
                input_data_dirs=config['input_dirs'],
                task_type=self.task_type,
                **config.get('loader_params', {})
            )
        elif self.data_dimension == '3d':
            return volumetric_data_loader(
                data_dir=config['data_dir'],
                task_type=self.task_type,
                **config.get('loader_params', {})
            )
        else:
            return enhanced_image_data_loader(
                data_dir=config['data_dir'],
                task_type=self.task_type,
                data_dimension=self.data_dimension,
                **config.get('loader_params', {})
            )

# Backward compatibility
custom_image_data_loader = enhanced_image_data_loader
