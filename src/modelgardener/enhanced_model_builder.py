"""
Enhanced Model Builder for ModelGardener

This module provides comprehensive model building capabilities with autom            if model_family.lower() == 'resnet':
                return self._build_resnet_model(model_name, input_shape, num_classes)
            elif model_family.lower() == 'efficientnet':
                return self._build_efficientnet_model(model_name, input_shape, num_classes)
            elif model_family.lower() == 'vgg':
                return self._build_vgg_model(model_name, input_shape, num_classes)
            elif model_family.lower() == 'densenet':
                return self._build_densenet_model(model_name, input_shape, num_classes)
            elif model_family.lower() == 'mobilenet':
                return self._build_mobilenet_model(model_name, input_shape, num_classes)
            elif model_family.lower() == 'cnn':
                return self._build_simple_cnn_model(model_name, input_shape, num_classes)
            else:
                raise ValueError(f"Unsupported model family: {model_family}")er detection, custom model support, and intelligent compilation.
"""

import tensorflow as tf
import keras
from typing import Dict, Any, Tuple
from .bridge_callback import BRIDGE


class EnhancedModelBuilder:
    """Enhanced model builder with automatic configuration and custom model support."""
    
    def __init__(self, config: Dict[str, Any], custom_functions: Dict[str, Any] = None):
        self.config = config
        self.custom_functions = custom_functions or {}
        self.model_config = config.get('model', {})
    
    def build_complete_model(self, input_shape: Tuple[int, ...], num_classes: int) -> keras.Model:
        """
        Build and compile a complete model.
        
        Args:
            input_shape: Input tensor shape (without batch dimension)
            num_classes: Number of output classes
            
        Returns:
            keras.Model: Compiled model ready for training
        """
        try:
            BRIDGE.log("=== Model Building ===")
            
            # Step 1: Build model architecture
            model = self._build_architecture(input_shape, num_classes)
            
            # Step 2: Compile model with optimizer, loss, and metrics
            compiled_model = self._compile_model(model)
            
            # Step 3: Display model information
            self._display_model_info(compiled_model)
            
            BRIDGE.log("Model building completed successfully")
            return compiled_model
            
        except Exception as e:
            BRIDGE.log(f"Error building model: {str(e)}")
            raise
    
    def _build_architecture(self, input_shape: Tuple[int, ...], num_classes: int) -> keras.Model:
        """Build model architecture based on configuration."""
        
        model_family = self.model_config.get('model_family', 'resnet')
        model_name = self.model_config.get('model_name', 'ResNet-50')
        
        BRIDGE.log(f"Building model: {model_family}/{model_name}")
        
        # Check if it's a custom model
        if model_family == 'custom_model':
            return self._build_custom_model(input_shape, num_classes)
        else:
            return self._build_builtin_model(model_family, model_name, input_shape, num_classes)
    
    def _build_custom_model(self, input_shape: Tuple[int, ...], num_classes: int) -> keras.Model:
        """Build custom model from user-defined functions."""
        
        model_name = self.model_config.get('model_name', '')
        custom_models = self.custom_functions.get('models', {})
        
        model_info = None
        for key, info in custom_models.items():
            if model_name in key or model_name == info.get('name', ''):
                model_info = info
                break
        
        if not model_info:
            raise ValueError(f"Custom model '{model_name}' not found in loaded functions")
        
        model_func = model_info['function']
        model_type = model_info['type']
        
        # Prepare model parameters
        model_params = self.model_config.get('model_parameters', {})
        model_params.update({
            'input_shape': input_shape,
            'num_classes': num_classes
        })
        
        try:
            if model_type == 'function':
                model = model_func(**model_params)
            elif model_type == 'class':
                model_instance = model_func(**model_params)
                if hasattr(model_instance, 'build'):
                    model = model_instance.build()
                elif hasattr(model_instance, '__call__'):
                    model = model_instance()
                else:
                    raise ValueError("Custom model class must have 'build' or '__call__' method")
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            if not isinstance(model, keras.Model):
                raise ValueError("Custom model function must return keras.Model instance")
            
            BRIDGE.log(f"Built custom model: {model_name}")
            return model
            
        except Exception as e:
            BRIDGE.log(f"Error building custom model: {str(e)}")
            raise
    
    def _build_builtin_model(self, model_family: str, model_name: str, 
                           input_shape: Tuple[int, ...], num_classes: int) -> keras.Model:
        """Build built-in model architectures."""
        
        try:
            if model_family.lower() == 'resnet':
                return self._build_resnet(model_name, input_shape, num_classes)
            elif model_family.lower() == 'efficientnet':
                return self._build_efficientnet(model_name, input_shape, num_classes)
            elif model_family.lower() == 'vgg':
                return self._build_vgg(model_name, input_shape, num_classes)
            elif model_family.lower() == 'densenet':
                return self._build_densenet(model_name, input_shape, num_classes)
            elif model_family.lower() == 'mobilenet':
                return self._build_mobilenet(model_name, input_shape, num_classes)
            else:
                raise ValueError(f"Unsupported model family: {model_family}")
                
        except Exception as e:
            BRIDGE.log(f"Error building {model_family} model: {str(e)}")
            raise
    
    def _build_simple_cnn_model(self, model_name: str, input_shape: Tuple[int, ...], num_classes: int):
        """Build a simple CNN model."""
        import tensorflow as tf
        from tensorflow import keras
        
        BRIDGE.log(f"Building simple CNN model: {model_name}")
        
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(num_classes, activation='softmax')
        ], name=model_name or 'SimpleCNN')
        
        BRIDGE.log(f"Simple CNN model created with input shape: {input_shape}, classes: {num_classes}")
        return model
    
    def _build_resnet(self, model_name: str, input_shape: Tuple[int, ...], num_classes: int) -> keras.Model:
        """Build ResNet model."""
        
        model_params = self.model_config.get('model_parameters', {})
        include_top = model_params.get('include_top', False)  # We'll add our own top
        weights = model_params.get('weights', 'imagenet')
        
        if model_name == 'ResNet-50':
            base_model = keras.applications.ResNet50(
                include_top=include_top,
                weights=weights,
                input_shape=input_shape
            )
        elif model_name == 'ResNet-101':
            base_model = keras.applications.ResNet101(
                include_top=include_top,
                weights=weights,
                input_shape=input_shape
            )
        elif model_name == 'ResNet-152':
            base_model = keras.applications.ResNet152(
                include_top=include_top,
                weights=weights,
                input_shape=input_shape
            )
        else:
            raise ValueError(f"Unsupported ResNet model: {model_name}")
        
        # Add custom classification head
        if not include_top:
            x = base_model.output
            x = keras.layers.GlobalAveragePooling2D()(x)
            x = keras.layers.Dense(512, activation='relu')(x)
            x = keras.layers.Dropout(0.5)(x)
            predictions = keras.layers.Dense(num_classes, activation='softmax')(x)
            
            model = keras.Model(inputs=base_model.input, outputs=predictions)
        else:
            model = base_model
        
        BRIDGE.log(f"Built ResNet model: {model_name}")
        return model
    
    def _build_efficientnet(self, model_name: str, input_shape: Tuple[int, ...], num_classes: int) -> keras.Model:
        """Build EfficientNet model."""
        
        model_params = self.model_config.get('model_parameters', {})
        include_top = model_params.get('include_top', False)
        weights = model_params.get('weights', 'imagenet')
        
        if model_name == 'EfficientNet-B0':
            base_model = keras.applications.EfficientNetB0(
                include_top=include_top,
                weights=weights,
                input_shape=input_shape
            )
        elif model_name == 'EfficientNet-B1':
            base_model = keras.applications.EfficientNetB1(
                include_top=include_top,
                weights=weights,
                input_shape=input_shape
            )
        elif model_name == 'EfficientNet-B2':
            base_model = keras.applications.EfficientNetB2(
                include_top=include_top,
                weights=weights,
                input_shape=input_shape
            )
        else:
            raise ValueError(f"Unsupported EfficientNet model: {model_name}")
        
        # Add custom classification head
        if not include_top:
            x = base_model.output
            x = keras.layers.GlobalAveragePooling2D()(x)
            x = keras.layers.Dense(256, activation='relu')(x)
            x = keras.layers.Dropout(0.3)(x)
            predictions = keras.layers.Dense(num_classes, activation='softmax')(x)
            
            model = keras.Model(inputs=base_model.input, outputs=predictions)
        else:
            model = base_model
        
        BRIDGE.log(f"Built EfficientNet model: {model_name}")
        return model
    
    def _build_vgg(self, model_name: str, input_shape: Tuple[int, ...], num_classes: int) -> keras.Model:
        """Build VGG model."""
        
        model_params = self.model_config.get('model_parameters', {})
        include_top = model_params.get('include_top', False)
        weights = model_params.get('weights', 'imagenet')
        
        if model_name == 'VGG-16':
            base_model = keras.applications.VGG16(
                include_top=include_top,
                weights=weights,
                input_shape=input_shape
            )
        elif model_name == 'VGG-19':
            base_model = keras.applications.VGG19(
                include_top=include_top,
                weights=weights,
                input_shape=input_shape
            )
        else:
            raise ValueError(f"Unsupported VGG model: {model_name}")
        
        # Add custom classification head
        if not include_top:
            x = base_model.output
            x = keras.layers.Flatten()(x)
            x = keras.layers.Dense(4096, activation='relu')(x)
            x = keras.layers.Dropout(0.5)(x)
            x = keras.layers.Dense(4096, activation='relu')(x)
            x = keras.layers.Dropout(0.5)(x)
            predictions = keras.layers.Dense(num_classes, activation='softmax')(x)
            
            model = keras.Model(inputs=base_model.input, outputs=predictions)
        else:
            model = base_model
        
        BRIDGE.log(f"Built VGG model: {model_name}")
        return model
    
    def _build_densenet(self, model_name: str, input_shape: Tuple[int, ...], num_classes: int) -> keras.Model:
        """Build DenseNet model."""
        
        model_params = self.model_config.get('model_parameters', {})
        include_top = model_params.get('include_top', False)
        weights = model_params.get('weights', 'imagenet')
        
        if model_name == 'DenseNet-121':
            base_model = keras.applications.DenseNet121(
                include_top=include_top,
                weights=weights,
                input_shape=input_shape
            )
        elif model_name == 'DenseNet-169':
            base_model = keras.applications.DenseNet169(
                include_top=include_top,
                weights=weights,
                input_shape=input_shape
            )
        elif model_name == 'DenseNet-201':
            base_model = keras.applications.DenseNet201(
                include_top=include_top,
                weights=weights,
                input_shape=input_shape
            )
        else:
            raise ValueError(f"Unsupported DenseNet model: {model_name}")
        
        # Add custom classification head
        if not include_top:
            x = base_model.output
            x = keras.layers.GlobalAveragePooling2D()(x)
            x = keras.layers.Dense(512, activation='relu')(x)
            x = keras.layers.Dropout(0.5)(x)
            predictions = keras.layers.Dense(num_classes, activation='softmax')(x)
            
            model = keras.Model(inputs=base_model.input, outputs=predictions)
        else:
            model = base_model
        
        BRIDGE.log(f"Built DenseNet model: {model_name}")
        return model
    
    def _build_mobilenet(self, model_name: str, input_shape: Tuple[int, ...], num_classes: int) -> keras.Model:
        """Build MobileNet model."""
        
        model_params = self.model_config.get('model_parameters', {})
        include_top = model_params.get('include_top', False)
        weights = model_params.get('weights', 'imagenet')
        
        if model_name == 'MobileNet':
            base_model = keras.applications.MobileNet(
                include_top=include_top,
                weights=weights,
                input_shape=input_shape
            )
        elif model_name == 'MobileNetV2':
            base_model = keras.applications.MobileNetV2(
                include_top=include_top,
                weights=weights,
                input_shape=input_shape
            )
        else:
            raise ValueError(f"Unsupported MobileNet model: {model_name}")
        
        # Add custom classification head
        if not include_top:
            x = base_model.output
            x = keras.layers.GlobalAveragePooling2D()(x)
            x = keras.layers.Dense(256, activation='relu')(x)
            x = keras.layers.Dropout(0.3)(x)
            predictions = keras.layers.Dense(num_classes, activation='softmax')(x)
            
            model = keras.Model(inputs=base_model.input, outputs=predictions)
        else:
            model = base_model
        
        BRIDGE.log(f"Built MobileNet model: {model_name}")
        return model
    
    def _compile_model(self, model: keras.Model) -> keras.Model:
        """Compile model with optimizer, loss, and metrics."""
        
        try:
            # Get optimizer
            optimizer = self._build_optimizer()
            
            # Get loss function
            loss_fn = self._build_loss_function()
            
            # Get metrics
            metrics = self._build_metrics()
            
            # Compile model
            model.compile(
                optimizer=optimizer,
                loss=loss_fn,
                metrics=metrics
            )
            
            BRIDGE.log("Model compiled successfully")
            return model
            
        except Exception as e:
            BRIDGE.log(f"Error compiling model: {str(e)}")
            raise
    
    def _build_optimizer(self):
        """Build optimizer from configuration."""
        
        optimizer_config = self.config.get('optimizer', {})
        selection_config = optimizer_config.get('Optimizer Selection', {})
        selected_optimizer = selection_config.get('selected_optimizer', 'Adam')
        
        # Check for custom optimizer
        custom_optimizers = self.custom_functions.get('optimizers', {})
        
        # First check if the optimizer is directly in custom functions (without Custom_ prefix)
        if selected_optimizer in custom_optimizers:
            optimizer_info = custom_optimizers[selected_optimizer]
            if isinstance(optimizer_info, dict):
                optimizer_func = optimizer_info.get('function')
                if optimizer_func is None:
                    optimizer_func = optimizer_info.get('loader')  # Alternative naming
                if optimizer_func:
                    return optimizer_func()
            else:
                return optimizer_info()
        
        # Then check with Custom_ prefix for backward compatibility
        if selected_optimizer.startswith('Custom_'):
            optimizer_info = custom_optimizers.get(selected_optimizer)
            if optimizer_info:
                if isinstance(optimizer_info, dict):
                    optimizer_func = optimizer_info.get('function')
                    if optimizer_func is None:
                        optimizer_func = optimizer_info.get('loader')  # Alternative naming
                    if optimizer_func:
                        return optimizer_func()
                else:
                    return optimizer_info()
        
        # Built-in optimizers
        training_config = self.config.get('training', {})
        learning_rate = training_config.get('initial_learning_rate', 0.001)
        
        # Override with optimizer-specific learning rate if available
        if 'learning_rate' in selection_config:
            learning_rate = selection_config['learning_rate']
        
        if selected_optimizer == 'Adam':
            beta_1 = selection_config.get('beta_1', 0.9)
            beta_2 = selection_config.get('beta_2', 0.999)
            epsilon = selection_config.get('epsilon', 1e-7)
            return keras.optimizers.Adam(
                learning_rate=learning_rate,
                beta_1=beta_1,
                beta_2=beta_2,
                epsilon=epsilon
            )
        elif selected_optimizer == 'SGD':
            momentum = selection_config.get('momentum', 0.9)
            nesterov = selection_config.get('nesterov', False)
            return keras.optimizers.SGD(
                learning_rate=learning_rate,
                momentum=momentum,
                nesterov=nesterov
            )
        elif selected_optimizer == 'RMSprop':
            rho = selection_config.get('rho', 0.9)
            momentum = selection_config.get('momentum', 0.0)
            return keras.optimizers.RMSprop(
                learning_rate=learning_rate,
                rho=rho,
                momentum=momentum
            )
        elif selected_optimizer == 'AdamW':
            weight_decay = selection_config.get('weight_decay', 0.01)
            return keras.optimizers.AdamW(
                learning_rate=learning_rate,
                weight_decay=weight_decay
            )
        else:
            # Default fallback
            BRIDGE.log(f"Unknown optimizer {selected_optimizer}, using Adam")
            return keras.optimizers.Adam(learning_rate=learning_rate)
    
    def _build_loss_function(self):
        """Build loss function from configuration."""
        
        loss_config = self.config.get('loss_functions', {})
        
        # Handle multi-output models
        output_config = loss_config.get('Model Output Configuration', {})
        num_outputs = output_config.get('num_outputs', 1)
        
        if num_outputs > 1:
            return self._build_multi_output_loss(loss_config, num_outputs)
        else:
            return self._build_single_loss(loss_config)
    
    def _build_single_loss(self, loss_config: Dict[str, Any]):
        """Build loss function for single output model."""
        
        selection_config = loss_config.get('Loss Selection', {})
        selected_loss = selection_config.get('selected_loss', 'Categorical Crossentropy')
        
        # Check for custom loss
        custom_losses = self.custom_functions.get('loss_functions', {})
        
        # First check if the selected loss is directly in custom functions (without Custom_ prefix)
        if selected_loss in custom_losses:
            loss_info = custom_losses[selected_loss]
            if isinstance(loss_info, dict):
                loss_func = loss_info.get('function')
                if loss_func is None:
                    loss_func = loss_info.get('loader')  # Alternative naming
                return loss_func
            else:
                return loss_info
        
        # Then check with Custom_ prefix for backward compatibility
        if selected_loss.startswith('Custom_'):
            loss_info = custom_losses.get(selected_loss)
            if loss_info:
                if isinstance(loss_info, dict):
                    loss_func = loss_info.get('function')
                    if loss_func is None:
                        loss_func = loss_info.get('loader')  # Alternative naming
                    return loss_func
                else:
                    return loss_info
        
        # Built-in losses
        if selected_loss == 'Categorical Crossentropy':
            return 'categorical_crossentropy'
        elif selected_loss == 'Sparse Categorical Crossentropy':
            return 'sparse_categorical_crossentropy'
        elif selected_loss == 'Binary Crossentropy':
            return 'binary_crossentropy'
        elif selected_loss == 'Mean Squared Error':
            return 'mse'
        elif selected_loss == 'Mean Absolute Error':
            return 'mae'
        else:
            BRIDGE.log(f"Unknown loss function {selected_loss}, using categorical_crossentropy")
            return 'categorical_crossentropy'
    
    def _build_multi_output_loss(self, loss_config: Dict[str, Any], num_outputs: int):
        """Build loss functions for multi-output model."""
        
        selection_config = loss_config.get('Loss Selection', {})
        output_config = loss_config.get('Model Output Configuration', {})
        loss_strategy = output_config.get('loss_strategy', 'single_loss_all_outputs')
        
        if loss_strategy == 'single_loss_all_outputs':
            # Same loss for all outputs
            return self._build_single_loss(loss_config)
        else:
            # Different loss for each output
            output_names = output_config.get('output_names', '').split(',')
            losses = {}
            
            for output_name in output_names:
                output_name = output_name.strip()
                if output_name in selection_config:
                    output_loss_config = {'Loss Selection': selection_config[output_name]}
                    losses[output_name] = self._build_single_loss(output_loss_config)
                else:
                    losses[output_name] = 'categorical_crossentropy'  # Default
            
            return losses
    
    def _build_metrics(self):
        """Build metrics from configuration."""
        
        metrics_config = self.config.get('metrics', {})
        
        # Handle multi-output models
        output_config = metrics_config.get('Model Output Configuration', {})
        num_outputs = output_config.get('num_outputs', 1)
        
        if num_outputs > 1:
            return self._build_multi_output_metrics(metrics_config, num_outputs)
        else:
            return self._build_single_metrics(metrics_config)
    
    def _build_single_metrics(self, metrics_config: Dict[str, Any]):
        """Build metrics for single output model."""
        
        selection_config = metrics_config.get('Metrics Selection', {})
        selected_metrics = selection_config.get('selected_metrics', 'Accuracy')
        
        if isinstance(selected_metrics, str):
            selected_metrics = [m.strip() for m in selected_metrics.split(',')]
        
        metrics_list = []
        
        # Check for custom metrics
        custom_metrics = self.custom_functions.get('metrics', {})
        
        for metric_name in selected_metrics:
            # First check if the metric is directly in custom functions (without Custom_ prefix)
            if metric_name in custom_metrics:
                metric_info = custom_metrics[metric_name]
                if isinstance(metric_info, dict):
                    metric_func = metric_info.get('function')
                    if metric_func is None:
                        metric_func = metric_info.get('loader')  # Alternative naming
                    if metric_func:
                        metrics_list.append(metric_func)
                else:
                    metrics_list.append(metric_info)
            elif metric_name.startswith('Custom_'):
                # Check with Custom_ prefix for backward compatibility
                metric_info = custom_metrics.get(metric_name)
                if metric_info:
                    if isinstance(metric_info, dict):
                        metric_func = metric_info.get('function')
                        if metric_func is None:
                            metric_func = metric_info.get('loader')  # Alternative naming
                        if metric_func:
                            metrics_list.append(metric_func)
                    else:
                        metrics_list.append(metric_info)
            else:
                # Built-in metrics
                if metric_name == 'Accuracy':
                    metrics_list.append('accuracy')
                elif metric_name == 'Categorical Accuracy':
                    metrics_list.append('categorical_accuracy')
                elif metric_name == 'Sparse Categorical Accuracy':
                    metrics_list.append('sparse_categorical_accuracy')
                elif metric_name == 'Top K Categorical Accuracy':
                    metrics_list.append('top_k_categorical_accuracy')
                elif metric_name == 'Precision':
                    metrics_list.append(keras.metrics.Precision())
                elif metric_name == 'Recall':
                    metrics_list.append(keras.metrics.Recall())
                elif metric_name == 'F1 Score':
                    metrics_list.append(keras.metrics.F1Score())
                elif metric_name == 'AUC':
                    metrics_list.append(keras.metrics.AUC())
        
        return metrics_list if metrics_list else ['accuracy']
    
    def _build_multi_output_metrics(self, metrics_config: Dict[str, Any], num_outputs: int):
        """Build metrics for multi-output model."""
        
        selection_config = metrics_config.get('Metrics Selection', {})
        output_config = metrics_config.get('Model Output Configuration', {})
        metrics_strategy = output_config.get('metrics_strategy', 'shared_metrics_all_outputs')
        
        if metrics_strategy == 'shared_metrics_all_outputs':
            # Same metrics for all outputs
            return self._build_single_metrics(metrics_config)
        else:
            # Different metrics for each output
            output_names = output_config.get('output_names', '').split(',')
            metrics = {}
            
            for output_name in output_names:
                output_name = output_name.strip()
                if output_name in selection_config:
                    output_metrics_config = {'Metrics Selection': selection_config[output_name]}
                    metrics[output_name] = self._build_single_metrics(output_metrics_config)
                else:
                    metrics[output_name] = ['accuracy']  # Default
            
            return metrics
    
    def _display_model_info(self, model: keras.Model):
        """Display model information."""
        
        try:
            # Log basic model info
            BRIDGE.log(f"Model: {model.name}")
            BRIDGE.log(f"Total parameters: {model.count_params():,}")
            
            # Count trainable parameters
            trainable_params = sum([tf.size(var).numpy() for var in model.trainable_variables])
            BRIDGE.log(f"Trainable parameters: {trainable_params:,}")
            
            # Log input/output shapes
            if hasattr(model, 'input_shape'):
                BRIDGE.log(f"Input shape: {model.input_shape}")
            
            if hasattr(model, 'output_shape'):
                if isinstance(model.output_shape, list):
                    for i, shape in enumerate(model.output_shape):
                        BRIDGE.log(f"Output {i+1} shape: {shape}")
                else:
                    BRIDGE.log(f"Output shape: {model.output_shape}")
            
        except Exception as e:
            BRIDGE.log(f"Error displaying model info: {str(e)}")
