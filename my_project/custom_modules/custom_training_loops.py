"""
Custom Training Loops Template for ModelGardener

This file provides templates for creating custom training loops.
These allow for advanced training strategies beyond standard fit() methods.
"""

import tensorflow as tf
import numpy as np
import time
from typing import Dict, Callable, Any, Optional


class GradualUnfreezingTrainer:
    """
    Custom trainer that gradually unfreezes layers during training.
    """
    
    def __init__(self, model, optimizer, loss_fn, unfreeze_schedule=None):
        """
        Args:
            model: Keras model to train
            optimizer: Optimizer to use
            loss_fn: Loss function
            unfreeze_schedule: Dict mapping epoch to number of layers to unfreeze
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.unfreeze_schedule = unfreeze_schedule or {0: 10, 5: 20, 10: -1}
        
        # Initially freeze all layers except the last few
        self._freeze_layers(10)
    
    def _freeze_layers(self, num_trainable):
        """Freeze all layers except the last num_trainable layers."""
        if num_trainable == -1:
            # Unfreeze all layers
            for layer in self.model.layers:
                layer.trainable = True
        else:
            # Freeze all first, then unfreeze last num_trainable
            for layer in self.model.layers:
                layer.trainable = False
            
            for layer in self.model.layers[-num_trainable:]:
                layer.trainable = True
    
    def train_step(self, batch_data):
        """Single training step."""
        images, labels = batch_data
        
        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            loss = self.loss_fn(labels, predictions)
        
        # Compute gradients and apply
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        return loss, predictions
    
    def train(self, dataset, epochs, validation_data=None, callbacks=None):
        """Custom training loop with gradual unfreezing."""
        history = {'loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Check if we need to unfreeze layers
            if epoch in self.unfreeze_schedule:
                self._freeze_layers(self.unfreeze_schedule[epoch])
                print(f"Epoch {epoch}: Unfroze {self.unfreeze_schedule[epoch]} layers")
            
            # Training loop
            epoch_loss = []
            for batch in dataset:
                loss, _ = self.train_step(batch)
                epoch_loss.append(loss)
            
            avg_loss = tf.reduce_mean(epoch_loss)
            history['loss'].append(float(avg_loss))
            
            # Validation
            if validation_data is not None:
                val_loss = self._validate(validation_data)
                history['val_loss'].append(float(val_loss))
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
            
            # Execute callbacks
            if callbacks:
                for callback in callbacks:
                    if hasattr(callback, 'on_epoch_end'):
                        callback.on_epoch_end(epoch, {'loss': avg_loss})
        
        return history
    
    def _validate(self, validation_data):
        """Validation step."""
        val_losses = []
        for val_batch in validation_data:
            val_images, val_labels = val_batch
            val_predictions = self.model(val_images, training=False)
            val_loss = self.loss_fn(val_labels, val_predictions)
            val_losses.append(val_loss)
        
        return tf.reduce_mean(val_losses)


class AdversarialTrainer:
    """
    Custom trainer for adversarial training.
    """
    
    def __init__(self, model, optimizer, loss_fn, epsilon=0.1, alpha=0.01):
        """
        Args:
            model: Model to train
            optimizer: Optimizer
            loss_fn: Loss function
            epsilon: Maximum perturbation magnitude
            alpha: Step size for adversarial examples
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epsilon = epsilon
        self.alpha = alpha
    
    def generate_adversarial_examples(self, images, labels):
        """Generate adversarial examples using FGSM."""
        with tf.GradientTape() as tape:
            tape.watch(images)
            predictions = self.model(images, training=True)
            loss = self.loss_fn(labels, predictions)
        
        # Compute gradients w.r.t. input
        gradients = tape.gradient(loss, images)
        
        # Generate adversarial examples
        signed_grad = tf.sign(gradients)
        adversarial_images = images + self.alpha * signed_grad
        adversarial_images = tf.clip_by_value(
            adversarial_images,
            images - self.epsilon,
            images + self.epsilon
        )
        adversarial_images = tf.clip_by_value(adversarial_images, 0.0, 1.0)
        
        return adversarial_images
    
    def train_step(self, batch_data):
        """Adversarial training step."""
        images, labels = batch_data
        
        # Generate adversarial examples
        adv_images = self.generate_adversarial_examples(images, labels)
        
        # Mix clean and adversarial examples
        mixed_images = tf.concat([images, adv_images], axis=0)
        mixed_labels = tf.concat([labels, labels], axis=0)
        
        # Training step
        with tf.GradientTape() as tape:
            predictions = self.model(mixed_images, training=True)
            loss = self.loss_fn(mixed_labels, predictions)
        
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        return loss


class CurriculumLearningTrainer:
    """
    Trainer implementing curriculum learning.
    """
    
    def __init__(self, model, optimizer, loss_fn, difficulty_fn=None):
        """
        Args:
            model: Model to train
            optimizer: Optimizer
            loss_fn: Loss function
            difficulty_fn: Function to determine sample difficulty
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.difficulty_fn = difficulty_fn or self._default_difficulty
    
    def _default_difficulty(self, sample, epoch):
        """Default difficulty function based on prediction confidence."""
        image, label = sample
        predictions = self.model(tf.expand_dims(image, 0), training=False)
        confidence = tf.reduce_max(tf.nn.softmax(predictions))
        return 1.0 - confidence  # Higher difficulty for low confidence
    
    def create_curriculum(self, dataset, epoch):
        """Create curriculum for current epoch."""
        # Collect all samples with difficulty scores
        samples_with_difficulty = []
        for sample in dataset.unbatch():
            difficulty = self._default_difficulty(sample, epoch)
            samples_with_difficulty.append((sample, difficulty))
        
        # Sort by difficulty (easy to hard)
        samples_with_difficulty.sort(key=lambda x: x[1])
        
        # Select subset based on epoch (gradually increase difficulty)
        total_samples = len(samples_with_difficulty)
        if epoch < 5:
            # Early epochs: use only easy samples
            selected_samples = samples_with_difficulty[:total_samples // 2]
        elif epoch < 15:
            # Middle epochs: use easy and medium samples
            selected_samples = samples_with_difficulty[:int(total_samples * 0.8)]
        else:
            # Later epochs: use all samples
            selected_samples = samples_with_difficulty
        
        # Extract just the samples
        curriculum_samples = [sample for sample, _ in selected_samples]
        
        return tf.data.Dataset.from_generator(
            lambda: curriculum_samples,
            output_signature=(
                tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        )


class MetaLearningTrainer:
    """
    Trainer for meta-learning (learning to learn).
    """
    
    def __init__(self, model, meta_optimizer, inner_optimizer, inner_steps=1):
        """
        Args:
            model: Model to meta-train
            meta_optimizer: Optimizer for meta-updates
            inner_optimizer: Optimizer for inner loop updates
            inner_steps: Number of inner loop steps
        """
        self.model = model
        self.meta_optimizer = meta_optimizer
        self.inner_optimizer = inner_optimizer
        self.inner_steps = inner_steps
    
    def inner_loop(self, support_data, loss_fn):
        """Inner loop adaptation."""
        # Create a copy of model weights
        initial_weights = [var.numpy() for var in self.model.trainable_variables]
        
        # Perform inner loop updates
        for _ in range(self.inner_steps):
            with tf.GradientTape() as tape:
                support_images, support_labels = support_data
                support_pred = self.model(support_images, training=True)
                support_loss = loss_fn(support_labels, support_pred)
            
            gradients = tape.gradient(support_loss, self.model.trainable_variables)
            self.inner_optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables)
            )
        
        return initial_weights
    
    def meta_step(self, task_batch, loss_fn):
        """Meta-learning step."""
        meta_gradients = []
        
        for task in task_batch:
            support_data, query_data = task
            
            # Save initial weights and perform inner loop
            initial_weights = self.inner_loop(support_data, loss_fn)
            
            # Compute meta-gradient on query set
            with tf.GradientTape() as tape:
                query_images, query_labels = query_data
                query_pred = self.model(query_images, training=True)
                query_loss = loss_fn(query_labels, query_pred)
            
            meta_grad = tape.gradient(query_loss, self.model.trainable_variables)
            meta_gradients.append(meta_grad)
            
            # Restore initial weights
            for var, initial_weight in zip(self.model.trainable_variables, initial_weights):
                var.assign(initial_weight)
        
        # Average meta-gradients and apply
        avg_meta_gradients = []
        for i in range(len(meta_gradients[0])):
            avg_grad = tf.reduce_mean([grad[i] for grad in meta_gradients], axis=0)
            avg_meta_gradients.append(avg_grad)
        
        self.meta_optimizer.apply_gradients(
            zip(avg_meta_gradients, self.model.trainable_variables)
        )


# Example usage and testing
if __name__ == "__main__":
    print("Testing custom training loops...")
    
    # Create dummy model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    optimizer = tf.keras.optimizers.Adam(0.001)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    
    # Test gradual unfreezing trainer
    trainer = GradualUnfreezingTrainer(model, optimizer, loss_fn)
    print(f"Gradual Unfreezing Trainer: {trainer.__class__.__name__}")
    
    # Test adversarial trainer
    adv_trainer = AdversarialTrainer(model, optimizer, loss_fn)
    print(f"Adversarial Trainer: {adv_trainer.__class__.__name__}")
    
    print("✅ Custom training loops template ready!")
