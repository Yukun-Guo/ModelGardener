"""
Scalable Dataset Loader for ModelGardener

This module provides enhanced dataset loading capabilities with generator patterns,
optimized tf.data pipelines, and support for large-scale datasets.
"""

import os
import tensorflow as tf
import numpy as np
from typing import Dict, Any, Optional, Callable, Tuple, List
from pathlib import Path
from bridge_callback import BRIDGE


class ScalableDatasetLoader:
    """Enhanced dataset loader with generator patterns and optimized pipelines."""
    
    def __init__(self, config: Dict[str, Any], custom_functions: Dict[str, Any] = None):
        self.config = config
        self.custom_functions = custom_functions or {}
        self.data_config = config.get('data', {})
        # Cache for datasets when custom loader returns both train/val
        self._cached_datasets = {}
    
    def load_dataset(self, split: str = 'train') -> tf.data.Dataset:
        """
        Main entry point for dataset loading.
        
        Args:
            split: Dataset split ('train' or 'val')
            
        Returns:
            tf.data.Dataset: Optimized dataset ready for training
        """
        try:
            BRIDGE.log(f"Loading {split} dataset...")
            
            # Step 1: Create base dataset
            dataset = self._create_base_dataset(split)
            
            # Step 2: Apply optimized pipeline
            dataset = self.create_optimized_pipeline(dataset, split)
            
            BRIDGE.log(f"{split.capitalize()} dataset loaded successfully")
            return dataset
            
        except Exception as e:
            BRIDGE.log(f"Error loading {split} dataset: {str(e)}")
            raise
    
    def _create_base_dataset(self, split: str) -> tf.data.Dataset:
        """Create base dataset using appropriate loader."""
        
        # Check for custom data loader
        data_loader_config = self.data_config.get('data_loader', {})
        selected_loader = data_loader_config.get('selected_data_loader', 'Default')
        
        if selected_loader.startswith('Custom_'):
            return self._create_custom_dataset(split, data_loader_config)
        else:
            return self._create_builtin_dataset(split)
    
    def _create_custom_dataset(self, split: str, loader_config: Dict[str, Any]) -> tf.data.Dataset:
        """Create dataset using custom data loader."""
        
        selected_loader = loader_config.get('selected_data_loader', '')
        custom_loader_info = self.custom_functions.get('data_loaders', {}).get(selected_loader)
        
        if not custom_loader_info:
            raise ValueError(f"Custom data loader {selected_loader} not found")
        
        # Check if we already have cached datasets from this loader
        cache_key = f"{selected_loader}_{hash(str(sorted(loader_config.items())))}"
        if cache_key in self._cached_datasets:
            BRIDGE.log(f"Using cached {split} dataset from custom loader: {selected_loader}")
            return self._cached_datasets[cache_key][split]
        
        loader_func = custom_loader_info['loader']
        loader_type = custom_loader_info['type']
        
        # Prepare arguments from configuration
        args = self._prepare_loader_args(loader_config, split)
        
        try:
            if loader_type == 'function':
                result = loader_func(**args)
                
                # Handle different return types
                if isinstance(result, tuple) and len(result) == 2:
                    # Custom loader returns (train_dataset, val_dataset)
                    train_ds, val_ds = result
                    
                    # Cache both datasets to avoid loading data twice
                    self._cached_datasets[cache_key] = {
                        'train': train_ds,
                        'val': val_ds
                    }
                    
                    dataset = train_ds if split == 'train' else val_ds
                else:
                    # Custom loader returns single dataset
                    dataset = result
                    
            elif loader_type == 'class':
                loader_instance = loader_func(**args)
                if hasattr(loader_instance, 'get_dataset'):
                    dataset = loader_instance.get_dataset(split)
                elif hasattr(loader_instance, '__call__'):
                    dataset = loader_instance(split)
                else:
                    raise ValueError("Custom loader class must have 'get_dataset' or '__call__' method")
            else:
                raise ValueError(f"Unknown loader type: {loader_type}")
                
            if not isinstance(dataset, tf.data.Dataset):
                raise ValueError("Custom loader must return tf.data.Dataset")
                
            BRIDGE.log(f"Loaded {split} dataset using custom loader: {selected_loader}")
            return dataset
            
        except Exception as e:
            BRIDGE.log(f"Error in custom data loader: {str(e)}")
            raise
    
    def _create_builtin_dataset(self, split: str) -> tf.data.Dataset:
        """Create dataset using built-in loaders."""
        
        # Get directory path
        dir_key = f"{split}_dir" if split in ['train', 'val'] else 'train_dir'
        data_dir = self.data_config.get(dir_key, '')
        
        if not data_dir or not os.path.exists(data_dir):
            # Try alternative directory naming
            alt_dir_key = f"{split}_data"
            data_dir = self.data_config.get(alt_dir_key, '')
            
            if not data_dir or not os.path.exists(data_dir):
                raise ValueError(f"Data directory not found: {data_dir}")
        
        # Get configuration parameters
        batch_size = self.data_config.get('batch_size', 32)
        image_size = self.data_config.get('image_size', [224, 224])
        if isinstance(image_size, int):
            image_size = [image_size, image_size]
        
        BRIDGE.log(f"Loading {split} dataset from: {data_dir}")
        
        # Use generator pattern for large datasets
        return self._create_generator_dataset(data_dir, split, batch_size, image_size)
    
    def _create_generator_dataset(self, data_dir: str, split: str, batch_size: int, image_size: List[int]) -> tf.data.Dataset:
        """Create dataset using generator pattern for memory efficiency."""
        
        try:
            # Find image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            image_files = []
            labels = []
            
            # Get class directories
            class_dirs = [d for d in Path(data_dir).iterdir() if d.is_dir()]
            class_names = sorted([d.name for d in class_dirs])
            
            if not class_names:
                raise ValueError(f"No class directories found in {data_dir}")
            
            # Create label mapping
            class_to_idx = {name: idx for idx, name in enumerate(class_names)}
            
            # Collect all image files and labels
            for class_dir in class_dirs:
                class_label = class_to_idx[class_dir.name]
                
                for ext in image_extensions:
                    for img_file in class_dir.glob(f"*{ext}"):
                        image_files.append(str(img_file))
                        labels.append(class_label)
            
            if not image_files:
                raise ValueError(f"No image files found in {data_dir}")
            
            BRIDGE.log(f"Found {len(image_files)} images in {len(class_names)} classes")
            
            # Create generator function
            def image_generator():
                for img_path, label in zip(image_files, labels):
                    yield img_path, label
            
            # Create dataset from generator
            dataset = tf.data.Dataset.from_generator(
                image_generator,
                output_signature=(
                    tf.TensorSpec(shape=(), dtype=tf.string),
                    tf.TensorSpec(shape=(), dtype=tf.int32)
                )
            )
            
            # Map to load and preprocess images
            dataset = dataset.map(
                lambda path, label: self._load_and_preprocess_image(path, label, image_size),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            
            return dataset
            
        except Exception as e:
            BRIDGE.log(f"Error creating generator dataset: {str(e)}")
            raise
    
    def _load_and_preprocess_image(self, image_path: tf.Tensor, label: tf.Tensor, image_size: List[int]) -> Tuple[tf.Tensor, tf.Tensor]:
        """Load and preprocess a single image."""
        
        # Load image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.cast(image, tf.float32)
        
        # Resize image
        image = tf.image.resize(image, image_size[:2])
        
        # Normalize to [0, 1]
        image = image / 255.0
        
        # Convert label to one-hot if needed
        # Note: This will be handled later in the pipeline
        
        return image, label
    
    def create_optimized_pipeline(self, dataset: tf.data.Dataset, split: str = 'train') -> tf.data.Dataset:
        """
        Create optimized tf.data pipeline with preprocessing and augmentation.
        
        Args:
            dataset: Base dataset
            split: Dataset split ('train' or 'val')
            
        Returns:
            tf.data.Dataset: Optimized dataset
        """
        try:
            BRIDGE.log(f"Creating optimized pipeline for {split} dataset")
            
            # Step 1: Apply preprocessing
            dataset = self._apply_preprocessing(dataset, split)
            
            # Step 2: Apply augmentation (only for training)
            if split == 'train':
                dataset = self._apply_augmentation(dataset)
            
            # Step 3: Performance optimizations
            dataset = self._apply_performance_optimizations(dataset, split)
            
            BRIDGE.log(f"Optimized pipeline created for {split} dataset")
            return dataset
            
        except Exception as e:
            BRIDGE.log(f"Error creating optimized pipeline: {str(e)}")
            raise
    
    def _apply_preprocessing(self, dataset: tf.data.Dataset, split: str) -> tf.data.Dataset:
        """Apply preprocessing transformations."""
        
        preprocessing_config = self.data_config.get('preprocessing', {})
        
        def preprocess_fn(images, labels):
            """
            Apply preprocessing to images and labels.
            Supports both single tensors and tuples of tensors for multi-input/output models.
            """
            def preprocess_single_image(image):
                """Apply preprocessing to a single image tensor."""
                # Apply normalization
                norm_config = preprocessing_config.get('Normalization', {})
                if norm_config.get('enabled', True):
                    method = norm_config.get('method', 'zero-center')
                    if method == 'zero-center':
                        # Normalize to [-1, 1]
                        image = (image - 0.5) * 2.0
                    elif method == 'standardize':
                        # Standardize using ImageNet statistics
                        mean = tf.constant([0.485, 0.456, 0.406])
                        std = tf.constant([0.229, 0.224, 0.225])
                        image = (image - mean) / std
                
                # Apply resizing if specified
                resize_config = preprocessing_config.get('Resizing', {})
                if resize_config.get('enabled', False):
                    target_size = resize_config.get('target_size', {})
                    width = target_size.get('width', 224)
                    height = target_size.get('height', 224)
                    image = tf.image.resize(image, [height, width])
                
                # Apply custom preprocessing functions
                image = self._apply_custom_preprocessing(image, preprocessing_config)
                
                return image
            
            # Handle multi-input case (tuple of images)
            if isinstance(images, tuple):
                processed_images = tuple(preprocess_single_image(img) for img in images)
            else:
                processed_images = preprocess_single_image(images)
            
            # Labels can be passed through as-is (preprocessing typically doesn't modify labels)
            # But we maintain the structure for multi-output support
            return processed_images, labels
        
        if preprocessing_config:
            dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
            BRIDGE.log("Applied preprocessing transformations")
        
        return dataset
    
    def _apply_custom_preprocessing(self, image: tf.Tensor, preprocessing_config: Dict[str, Any]) -> tf.Tensor:
        """Apply custom preprocessing functions to an image."""
        
        # Get custom preprocessing functions
        custom_preprocessing = self.custom_functions.get('preprocessing', {})
        if not custom_preprocessing:
            # Check alternative naming
            custom_preprocessing = self.custom_functions.get('preprocessing_functions', {})
        
        if not custom_preprocessing:
            return image
        
        # Apply each enabled custom preprocessing function
        for func_name, func_info in custom_preprocessing.items():
            # Check if this custom function is enabled in config
            func_config = preprocessing_config.get(func_name, {})
            if not func_config.get('enabled', False):
                continue
            
            try:
                # Get the function
                if isinstance(func_info, dict):
                    func = func_info.get('function')
                    if func is None:
                        func = func_info.get('loader')  # Alternative naming
                else:
                    func = func_info
                
                if func is None:
                    BRIDGE.log(f"Warning: Custom preprocessing function {func_name} not found")
                    continue
                
                # Prepare function parameters
                func_params = {}
                if isinstance(func_info, dict) and 'parameters' in func_info:
                    for param in func_info['parameters']:
                        param_name = param['name']
                        param_value = func_config.get(param_name, param.get('default'))
                        func_params[param_name] = param_value
                
                # Apply the custom function
                if func_params:
                    image = func(image, **func_params)
                else:
                    image = func(image)
                
                BRIDGE.log(f"Applied custom preprocessing: {func_name}")
                
            except Exception as e:
                BRIDGE.log(f"Error applying custom preprocessing {func_name}: {str(e)}")
                # Continue with other preprocessing steps
                continue
        
        return image
    
    def _apply_augmentation(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Apply data augmentation for training dataset."""
        
        aug_config = self.data_config.get('augmentation', {})
        
        def augment_fn(images, labels):
            """
            Apply data augmentation to images and labels.
            Supports both single tensors and tuples of tensors for multi-input/output models.
            """
            def augment_single_image(image):
                """Apply augmentation to a single image tensor."""
                # Horizontal flip
                hflip_config = aug_config.get('Horizontal Flip', {})
                if hflip_config.get('enabled', False):
                    prob = hflip_config.get('probability', 0.5)
                    image = tf.cond(
                        tf.random.uniform([]) < prob,
                        lambda: tf.image.flip_left_right(image),
                        lambda: image
                    )
                
                # Vertical flip
                vflip_config = aug_config.get('Vertical Flip', {})
                if vflip_config.get('enabled', False):
                    prob = vflip_config.get('probability', 0.5)
                    image = tf.cond(
                        tf.random.uniform([]) < prob,
                        lambda: tf.image.flip_up_down(image),
                        lambda: image
                    )
                
                # Random rotation
                rotation_config = aug_config.get('Rotation', {})
                if rotation_config.get('enabled', False):
                    angle_range = rotation_config.get('angle_range', 15.0)
                    prob = rotation_config.get('probability', 0.5)
                    
                    angle = tf.random.uniform([], -angle_range, angle_range) * (3.14159265359 / 180.0)
                    image = tf.cond(
                        tf.random.uniform([]) < prob,
                        lambda: tf.image.rot90(image, k=tf.cast(angle / (3.14159265359/2), tf.int32)),
                        lambda: image
                    )
                
                # Brightness adjustment
                brightness_config = aug_config.get('Brightness', {})
                if brightness_config.get('enabled', False):
                    delta_range = brightness_config.get('delta_range', 0.2)
                    prob = brightness_config.get('probability', 0.5)
                    
                    image = tf.cond(
                        tf.random.uniform([]) < prob,
                        lambda: tf.image.random_brightness(image, delta_range),
                        lambda: image
                    )
                
                # Contrast adjustment
                contrast_config = aug_config.get('Contrast', {})
                if contrast_config.get('enabled', False):
                    factor_range = contrast_config.get('factor_range', [0.8, 1.2])
                    prob = contrast_config.get('probability', 0.5)
                    
                    image = tf.cond(
                        tf.random.uniform([]) < prob,
                        lambda: tf.image.random_contrast(image, factor_range[0], factor_range[1]),
                        lambda: image
                    )
                
                # Gaussian noise
                noise_config = aug_config.get('Gaussian Noise', {})
                if noise_config.get('enabled', False):
                    std_dev = noise_config.get('std_dev', 0.1)
                    prob = noise_config.get('probability', 0.5)
                    
                    noise = tf.random.normal(tf.shape(image), 0.0, std_dev)
                    image = tf.cond(
                        tf.random.uniform([]) < prob,
                        lambda: tf.clip_by_value(image + noise, 0.0, 1.0),
                        lambda: image
                    )
                
                # Apply custom augmentation functions
                image = self._apply_custom_augmentations(image, aug_config)
                
                # Ensure image values are in valid range
                image = tf.clip_by_value(image, 0.0, 1.0)
                return image
            
            # Handle multi-input case (tuple of images)
            if isinstance(images, tuple):
                augmented_images = tuple(augment_single_image(img) for img in images)
            else:
                augmented_images = augment_single_image(images)
            
            # Labels typically don't need augmentation, but we maintain the structure
            return augmented_images, labels
        
        if aug_config:
            dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
            BRIDGE.log("Applied data augmentation")
        
        return dataset
    
    def _apply_custom_augmentations(self, image: tf.Tensor, aug_config: Dict[str, Any]) -> tf.Tensor:
        """Apply custom augmentation functions to an image."""
        
        # Get custom augmentation functions
        custom_augmentations = self.custom_functions.get('augmentations', {})
        if not custom_augmentations:
            # Check alternative naming
            custom_augmentations = self.custom_functions.get('augmentation_functions', {})
        
        if not custom_augmentations:
            return image
        
        # Apply each enabled custom augmentation function
        for func_name, func_info in custom_augmentations.items():
            # Check if this custom function is enabled in config
            func_config = aug_config.get(func_name, {})
            if not func_config.get('enabled', False):
                continue
            
            try:
                # Get the function
                if isinstance(func_info, dict):
                    func = func_info.get('function')
                    if func is None:
                        func = func_info.get('loader')  # Alternative naming
                else:
                    func = func_info
                
                if func is None:
                    BRIDGE.log(f"Warning: Custom augmentation function {func_name} not found")
                    continue
                
                # Check probability for stochastic application
                probability = func_config.get('probability', 1.0)
                should_apply = tf.random.uniform([]) < probability
                
                # Prepare function parameters
                func_params = {}
                if isinstance(func_info, dict) and 'parameters' in func_info:
                    for param in func_info['parameters']:
                        param_name = param['name']
                        param_value = func_config.get(param_name, param.get('default'))
                        func_params[param_name] = param_value
                
                # Apply the custom function conditionally
                def apply_custom_aug():
                    if func_params:
                        return func(image, **func_params)
                    else:
                        return func(image)
                
                image = tf.cond(should_apply, apply_custom_aug, lambda: image)
                
                BRIDGE.log(f"Applied custom augmentation: {func_name}")
                
            except Exception as e:
                BRIDGE.log(f"Error applying custom augmentation {func_name}: {str(e)}")
                # Continue with other augmentation steps
                continue
        
        return image
    
    def _apply_performance_optimizations(self, dataset: tf.data.Dataset, split: str) -> tf.data.Dataset:
        """Apply performance optimizations to the dataset."""
        
        batch_size = self.data_config.get('batch_size', 32)
        
        # Cache dataset if it's small enough
        cache_config = self.data_config.get('cache', {})
        if cache_config.get('enabled', False):
            cache_file = cache_config.get('cache_file', '')
            if cache_file:
                dataset = dataset.cache(cache_file)
            else:
                dataset = dataset.cache()
            BRIDGE.log("Dataset caching enabled")
        
        # Shuffle training data
        if split == 'train':
            shuffle_config = self.data_config.get('shuffle', {})
            buffer_size = shuffle_config.get('buffer_size', 1000)
            dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)
            BRIDGE.log(f"Applied shuffling with buffer size: {buffer_size}")
        
        # Batch the dataset (skip for custom loaders that already return batched data)
        data_loader_config = self.data_config.get('data_loader', {})
        selected_loader = data_loader_config.get('selected_data_loader', 'Default')
        
        if not selected_loader.startswith('Custom_'):
            # Only batch built-in loaders, custom loaders may already be batched
            dataset = dataset.batch(batch_size, drop_remainder=False)
        
        # Convert labels to categorical if needed (skip for custom loaders that already return categorical)
        # Only convert labels for built-in loaders, custom loaders may already return categorical
        if not selected_loader.startswith('Custom_'):
            dataset = dataset.map(
                self._convert_labels_to_categorical,
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        # Prefetch for performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        BRIDGE.log(f"Applied performance optimizations (batch_size: {batch_size})")
        return dataset
    
    def _convert_labels_to_categorical(self, images, labels):
        """Convert labels to categorical format if needed. Supports multi-output models."""
        
        def convert_single_label(single_labels):
            """Convert a single label tensor to categorical format."""
            # Get number of classes from config or infer
            num_classes = self.data_config.get('num_classes')
            
            if num_classes is None:
                # Try to infer from data
                max_label = tf.reduce_max(single_labels)
                num_classes = max_label + 1
            
            # Convert to one-hot encoding
            labels_categorical = tf.one_hot(tf.cast(single_labels, tf.int32), depth=tf.cast(num_classes, tf.int32))
            return labels_categorical
        
        # Handle multi-output case (tuple of labels)
        if isinstance(labels, tuple):
            converted_labels = tuple(convert_single_label(lbl) for lbl in labels)
        else:
            converted_labels = convert_single_label(labels)
        
        return images, converted_labels
    
    def _prepare_loader_args(self, loader_config: Dict[str, Any], split: str) -> Dict[str, Any]:
        """Prepare arguments for custom data loader."""
        args = {}
        
        # Add common parameters
        dir_key = f"{split}_dir" if split in ['train', 'val'] else 'train_dir'
        data_dir = self.data_config.get(dir_key, '')
        
        args['data_dir'] = data_dir
        args['split'] = split
        
        # Add loader-specific parameters from config
        parameters = loader_config.get('parameters', {})
        
        if isinstance(parameters, dict):
            for key, value in parameters.items():
                args[key] = value
        
        return args
    
    def infer_data_specs(self):
        """
        Infer input shape and number of classes from the dataset.
        Supports both single and multi-input/output models.
        
        Returns:
            For single input/output: Tuple[Tuple[int, ...], int]: (input_shape, num_classes)
            For multi-input/output: Tuple[Tuple[Tuple[int, ...], ...], Tuple[int, ...]]: (input_shapes, num_classes_list)
        """
        try:
            # Create a small sample dataset to infer specs
            sample_dataset = self._create_base_dataset('train')
            sample_dataset = sample_dataset.take(1)
            
            for batch in sample_dataset:
                if isinstance(batch, tuple) and len(batch) == 2:
                    images, labels = batch
                    
                    # Handle multi-input case
                    if isinstance(images, tuple):
                        input_shapes = tuple(tuple(img.shape[1:]) for img in images)  # Remove batch dimension
                        BRIDGE.log(f"Inferred multi-input shapes: {input_shapes}")
                    else:
                        input_shapes = tuple(images.shape[1:])  # Remove batch dimension
                        BRIDGE.log(f"Inferred single input shape: {input_shapes}")
                    
                    # Handle multi-output case
                    if isinstance(labels, tuple):
                        num_classes_list = []
                        for lbl in labels:
                            if len(lbl.shape) > 1:
                                # One-hot encoded
                                num_classes_list.append(lbl.shape[-1])
                            else:
                                # Sparse labels
                                num_classes_list.append(tf.reduce_max(lbl).numpy() + 1)
                        num_classes = tuple(num_classes_list)
                        BRIDGE.log(f"Inferred multi-output classes: {num_classes}")
                    else:
                        # Single output
                        if len(labels.shape) > 1:
                            # One-hot encoded
                            num_classes = labels.shape[-1]
                        else:
                            # Sparse labels
                            num_classes = tf.reduce_max(labels).numpy() + 1
                        BRIDGE.log(f"Inferred single output classes: {num_classes}")
                    
                    return input_shapes, int(num_classes) if not isinstance(num_classes, tuple) else tuple(int(nc) for nc in num_classes)
            
        except Exception as e:
            BRIDGE.log(f"Error inferring data specs: {str(e)}")
        
        # Fallback to configuration defaults
        image_size = self.data_config.get('image_size', [224, 224])
        if isinstance(image_size, int):
            image_size = [image_size, image_size]
        
        input_shape = tuple(image_size + [3])  # Assume RGB
        num_classes = self.data_config.get('num_classes', 1000)
        
        BRIDGE.log(f"Using default input shape: {input_shape}")
        BRIDGE.log(f"Using default number of classes: {num_classes}")
        
        return input_shape, num_classes
