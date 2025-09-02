import keras

def adaptive_adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, 
                 epsilon=1e-7, decay_factor=0.99):
    """
    Custom Adam optimizer with adaptive learning rate decay.
    
    Args:
        learning_rate: Initial learning rate
        beta_1: Exponential decay rate for first moment estimates
        beta_2: Exponential decay rate for second moment estimates
        epsilon: Small constant for numerical stability
        decay_factor: Factor for exponential learning rate decay
    
    Returns:
        TensorFlow optimizer instance
    """
    
    # Create exponential decay schedule
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=1000,
        decay_rate=decay_factor,
        staircase=True
    )
    
    return keras.optimizers.Adam(
        learning_rate=lr_schedule,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        name="AdaptiveAdam"
    )

