"""
Custom Augmentations Template for ModelGardener

This file provides templates for creating custom data augmentation functions.
Augmentation functions should work with tf.data.Dataset objects.
"""

import tensorflow as tf
import numpy as np


def random_rotation_3d(image, max_angle=30):
    """
    Apply random 3D rotation to an image.
    
    Args:
        image: Input image tensor
        max_angle: Maximum rotation angle in degrees
        
    Returns:
        Rotated image
    """
    angle = tf.random.uniform([], -max_angle, max_angle) * np.pi / 180
    return tf.image.rot90(image, k=tf.cast(angle / (np.pi / 2), tf.int32))


def random_brightness_contrast(image, brightness_delta=0.2, contrast_delta=0.2):
    """
    Apply random brightness and contrast adjustments.
    
    Args:
        image: Input image tensor
        brightness_delta: Maximum brightness change
        contrast_delta: Maximum contrast change
        
    Returns:
        Adjusted image
    """
    # Random brightness
    image = tf.image.random_brightness(image, brightness_delta)
    
    # Random contrast
    image = tf.image.random_contrast(image, 1 - contrast_delta, 1 + contrast_delta)
    
    # Clip values to valid range
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image


def random_cutout(image, mask_size=32, num_masks=1):
    """
    Apply random cutout (erasing) to an image.
    
    Args:
        image: Input image tensor
        mask_size: Size of the square mask
        num_masks: Number of masks to apply
        
    Returns:
        Image with random cutouts
    """
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    
    for _ in range(num_masks):
        # Random position
        y = tf.random.uniform([], 0, height - mask_size, dtype=tf.int32)
        x = tf.random.uniform([], 0, width - mask_size, dtype=tf.int32)
        
        # Create mask
        mask = tf.ones([mask_size, mask_size, tf.shape(image)[2]])
        
        # Apply cutout
        image = tf.tensor_scatter_nd_update(
            image,
            [[y + i, x + j, c] for i in range(mask_size) for j in range(mask_size) for c in range(tf.shape(image)[2])],
            tf.zeros([mask_size * mask_size * tf.shape(image)[2]])
        )
    
    return image


def mixup_augmentation(image1, label1, image2, label2, alpha=0.2):
    """
    Apply mixup augmentation to two images.
    
    Args:
        image1, image2: Input images
        label1, label2: Corresponding labels
        alpha: Mixup parameter
        
    Returns:
        Mixed image and label
    """
    # Sample lambda from Beta distribution
    lam = tf.random.uniform([], 0, 1)
    if alpha > 0:
        lam = tf.random.gamma([], alpha, alpha)
        lam = lam / (lam + tf.random.gamma([], alpha, alpha))
    
    # Mix images
    mixed_image = lam * image1 + (1 - lam) * image2
    
    # Mix labels
    mixed_label = lam * label1 + (1 - lam) * label2
    
    return mixed_image, mixed_label


def cutmix_augmentation(image1, label1, image2, label2, alpha=1.0):
    """
    Apply CutMix augmentation to two images.
    
    Args:
        image1, image2: Input images
        label1, label2: Corresponding labels
        alpha: CutMix parameter
        
    Returns:
        CutMix image and label
    """
    height, width = tf.shape(image1)[0], tf.shape(image1)[1]
    
    # Sample lambda
    lam = tf.random.uniform([], 0, 1)
    if alpha > 0:
        lam = tf.random.gamma([], alpha, alpha)
        lam = lam / (lam + tf.random.gamma([], alpha, alpha))
    
    # Calculate cut size
    cut_ratio = tf.sqrt(1 - lam)
    cut_w = tf.cast(width * cut_ratio, tf.int32)
    cut_h = tf.cast(height * cut_ratio, tf.int32)
    
    # Random position
    cx = tf.random.uniform([], 0, width, dtype=tf.int32)
    cy = tf.random.uniform([], 0, height, dtype=tf.int32)
    
    # Calculate box coordinates
    bbx1 = tf.clip_by_value(cx - cut_w // 2, 0, width)
    bby1 = tf.clip_by_value(cy - cut_h // 2, 0, height)
    bbx2 = tf.clip_by_value(cx + cut_w // 2, 0, width)
    bby2 = tf.clip_by_value(cy + cut_h // 2, 0, height)
    
    # Create mask
    mask = tf.ones_like(image1)
    mask = tf.tensor_scatter_nd_update(
        mask,
        [[i, j, c] for i in range(bby1, bby2) for j in range(bbx1, bbx2) for c in range(tf.shape(image1)[2])],
        tf.zeros([(bby2 - bby1) * (bbx2 - bbx1) * tf.shape(image1)[2]])
    )
    
    # Apply CutMix
    mixed_image = image1 * mask + image2 * (1 - mask)
    
    # Adjust lambda based on actual cut area
    lam = 1 - tf.cast((bbx2 - bbx1) * (bby2 - bby1), tf.float32) / tf.cast(width * height, tf.float32)
    mixed_label = lam * label1 + (1 - lam) * label2
    
    return mixed_image, mixed_label


def random_perspective_transform(image, distortion_scale=0.5):
    """
    Apply random perspective transformation.
    
    Args:
        image: Input image tensor
        distortion_scale: Scale of perspective distortion
        
    Returns:
        Transformed image
    """
    height, width = tf.cast(tf.shape(image)[0], tf.float32), tf.cast(tf.shape(image)[1], tf.float32)
    
    # Original corners
    src_corners = tf.constant([[0., 0.], [width, 0.], [width, height], [0., height]])
    
    # Add random distortion to corners
    distortion = tf.random.uniform([4, 2], -distortion_scale, distortion_scale)
    dst_corners = src_corners + distortion * tf.stack([width, height])
    
    # Apply perspective transform (simplified version)
    # Note: In practice, you might want to use a more sophisticated transform
    return tf.image.resize(image, [tf.shape(image)[0], tf.shape(image)[1]])


def color_jittering(image, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
    """
    Apply color jittering with multiple color adjustments.
    
    Args:
        image: Input image tensor
        brightness: Brightness adjustment range
        contrast: Contrast adjustment range
        saturation: Saturation adjustment range
        hue: Hue adjustment range
        
    Returns:
        Color-adjusted image
    """
    # Apply transformations in random order
    transforms = [
        lambda img: tf.image.random_brightness(img, brightness),
        lambda img: tf.image.random_contrast(img, 1 - contrast, 1 + contrast),
        lambda img: tf.image.random_saturation(img, 1 - saturation, 1 + saturation),
        lambda img: tf.image.random_hue(img, hue)
    ]
    
    # Shuffle and apply transforms
    for transform in transforms:
        if tf.random.uniform([]) > 0.5:
            image = transform(image)
    
    # Clip to valid range
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image


class AugmentationPipeline:
    """
    Class for creating complex augmentation pipelines.
    """
    
    def __init__(self, augmentations, probabilities=None):
        """
        Args:
            augmentations: List of augmentation functions
            probabilities: List of probabilities for each augmentation
        """
        self.augmentations = augmentations
        self.probabilities = probabilities or [0.5] * len(augmentations)
    
    def __call__(self, image, label=None):
        """Apply random augmentations from the pipeline."""
        for aug, prob in zip(self.augmentations, self.probabilities):
            if tf.random.uniform([]) < prob:
                if label is not None:
                    # For augmentations that modify both image and label
                    try:
                        image, label = aug(image, label)
                    except:
                        image = aug(image)
                else:
                    image = aug(image)
        
        return (image, label) if label is not None else image


# Example usage and testing
if __name__ == "__main__":
    print("Testing custom augmentations...")
    
    # Create dummy image
    dummy_image = tf.random.uniform([224, 224, 3], 0, 1, dtype=tf.float32)
    
    # Test brightness/contrast
    aug_image = random_brightness_contrast(dummy_image)
    print(f"Brightness/Contrast: {aug_image.shape}")
    
    # Test color jittering
    aug_image = color_jittering(dummy_image)
    print(f"Color Jittering: {aug_image.shape}")
    
    # Test augmentation pipeline
    pipeline = AugmentationPipeline([
        random_brightness_contrast,
        color_jittering
    ])
    aug_image = pipeline(dummy_image)
    print(f"Pipeline: {aug_image.shape}")
    
    print("✅ Custom augmentations template ready!")
