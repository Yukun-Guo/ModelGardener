"""
Custom callback functions and classes for ModelGardener.

These callbacks demonstrate how to create custom training callbacks that can be 
dynamically loaded into the callbacks parameter tree. Callbacks must be classes that inherit from tf.keras.callbacks.Callback

CallBack class requirements:
- Should implement relevant callback methods (on_epoch_end, on_batch_end, etc.)
- __init__ method parameters become configuration options
"""

import keras

class ExampleCallbackClass1(keras.callbacks.Callback):
    """
    Example custom callback class.
    """
    
    def __init__(self,
                 param1: int = 1,
                 param2: float = 0.1):
        """
        Initialize the callback with configuration parameters.
        Args:
            param1 (int): Example integer parameter.
            param2 (float): Example float parameter.
        """
        super().__init__()
        self.param1 = param1
        self.param2 = param2
        
    def on_epoch_end(self, epoch, logs=None):
        # Add custom logic to execute at the end of each epoch
        # print(f"Example Callback Class 1: Epoch {epoch} ended. Param1: {self.param1}, Param2: {self.param2}")
        return super().on_epoch_end(epoch, logs)
    
class ExampleCallbackClass2(keras.callbacks.Callback):
    """
    Another example custom callback class.
    """
    
    def __init__(self,
                 param1: str = "default",
                 param2: bool = True):
        """
        Initialize the callback with configuration parameters.
        Args:
            param1 (str): Example string parameter.
            param2 (bool): Example boolean parameter.
        """
        super().__init__()
        self.param1 = param1
        self.param2 = param2

    def on_batch_end(self, batch, logs=None):
        # Add custom logic to execute at the end of each batch
        # print(f"Example Callback Class 2: Batch {batch} ended. Param1: {self.param1}, Param2: {self.param2}")
        return super().on_batch_end(batch, logs)
