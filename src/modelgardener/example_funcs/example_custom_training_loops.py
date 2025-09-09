"""
Example custom training loop functions for ModelGardener

Functions should follow the nested wrapper pattern.

This file demonstrates how to create custom training loop functions that can be
dynamically loaded into the ModelGardener application. Training loops must be
functions that implement custom training logic:
- Should accept parameters like model, optimizer, loss, data, epochs, etc.
- Should implement the complete training loop logic
- Should handle both training and validation phases
"""
import tensorflow as tf

def example_training_loop(param1=1, param2=1):
    
    def wrapper(model, train_dataset, val_dataset=None, epochs=100, optimizer=None, loss_fn=None):
        """
        Example custom training loop function.
        
        Args:
            model: Keras model to be trained
            train_dataset: Training dataset (tf.data.Dataset)
            val_dataset: Validation dataset (tf.data.Dataset), optional
            epochs: Number of training epochs
            optimizer: Keras optimizer instance
            loss_fn: Loss function
        
        Returns:
            Trained Keras model
        """
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            # Training phase
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    logits = model(x_batch_train, training=True)
                    loss_value = loss_fn(y_batch_train, logits)
                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply(zip(grads, model.trainable_weights))
                
                if step % 100 == 0:
                    print(f"Training loss at step {step}: {loss_value:.4f}")
            
            # Validation phase
            if val_dataset is not None:
                val_loss = 0
                val_steps = 0
                for x_batch_val, y_batch_val in val_dataset:
                    val_logits = model(x_batch_val, training=False)
                    val_loss += loss_fn(y_batch_val, val_logits)
                    val_steps += 1
                val_loss /= val_steps
                print(f"Validation loss after epoch {epoch+1}: {val_loss:.4f}")
        
        return wrapper