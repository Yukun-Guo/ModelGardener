import os
import threading
import copy
from typing import Dict, Any
from bridge_callback import BRIDGE
from official.core import  train_lib, exp_factory
# ---------------------------
# Trainer thread that calls train_lib.run_experiment
# ---------------------------

# ---------------------------
# Helper: map GUI config -> exp_config
# This is a defensive mapping because different experiments expect different fields.
# You should adapt this function to match the exact exp_name you will use.
# ---------------------------




class TFModelsTrainerThread(threading.Thread):
    def __init__(self, gui_cfg: Dict[str, Any], exp_name: str = "image_classification_imagenet", resume_ckpt: str = None):
        super().__init__()
        self.gui_cfg = copy.deepcopy(gui_cfg)
        self.exp_name = exp_name
        self.resume_ckpt = resume_ckpt
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):

        try:
            BRIDGE.log.emit(f"Building exp_config for '{self.exp_name}' ...")
            exp_cfg = self.map_gui_to_expconfig(self.gui_cfg, self.exp_name)

            # set init_checkpoint if resume path provided
            if self.resume_ckpt:
                try:
                    exp_cfg.task.init_checkpoint = self.resume_ckpt
                    BRIDGE.log.emit(f"Set init_checkpoint: {self.resume_ckpt}")
                except Exception:
                    pass

            # ensure model_dir using new config structure
            runtime_cfg = self.gui_cfg.get("runtime", {})
            model_dir = runtime_cfg.get("model_dir", "./model_dir")
            try:
                exp_cfg.runtime.model_dir = model_dir
            except Exception:
                try:
                    exp_cfg.runtime = exp_cfg.get("runtime", {})
                    exp_cfg.runtime.model_dir = model_dir
                except Exception:
                    pass
            os.makedirs(model_dir, exist_ok=True)

            # Add our QtBridgeCallback - get epochs from training config
            training_cfg = self.gui_cfg.get("training", {})
            total_steps = int(training_cfg.get("epochs", 1))
            cb = {"type": "QtBridgeCallback", "total_train_steps": total_steps, "log_every_n": 1}
            # ensure callbacks list exists
            try:
                if hasattr(exp_cfg, "callbacks") and exp_cfg.callbacks:
                    # remove previous QtBridgeCallback entries
                    exp_cfg.callbacks = [c for c in exp_cfg.callbacks if not (isinstance(c, dict) and c.get("type") == "QtBridgeCallback")]
                    exp_cfg.callbacks.append(cb)
                else:
                    exp_cfg.callbacks = [cb]
            except Exception:
                try:
                    exp_cfg.callbacks = [cb]
                except Exception:
                    pass

            BRIDGE.log.emit("Starting train_lib.run_experiment ...")
            # Note: some run_experiment wrappers accept distribution_strategy as arg; we pass runtime setting if present
            ds = None
            distribution = None
            try:
                distribution = getattr(exp_cfg.runtime, "distribution_strategy", None)
            except Exception:
                distribution = None

            # Run training. This is a blocking call.
            train_lib.run_experiment(
                distribution_strategy=distribution or "mirrored",
                mode="train",
                params=exp_cfg
            )

            BRIDGE.log.emit("train_lib.run_experiment returned (training finished).")
        except Exception as e:
            BRIDGE.log.emit(f"Training exception: {e}")
        finally:
            BRIDGE.finished.emit()

    def map_gui_to_expconfig(self, gui_cfg: Dict[str, Any], exp_name: str):
        """
        Returns a ConfigDict from exp_factory.get_exp_config(exp_name) with fields
        updated from comprehensive gui_cfg structure.
        """

        exp_cfg = exp_factory.get_exp_config(exp_name)  # ConfigDict

        # Map basic configuration
        try:
            # Runtime settings
            if 'runtime' in gui_cfg:
                runtime = gui_cfg['runtime']
                exp_cfg.runtime.model_dir = runtime.get('model_dir', './model_dir')
                if runtime.get('distribution_strategy'):
                    exp_cfg.runtime.distribution_strategy = runtime['distribution_strategy']
                if runtime.get('mixed_precision'):
                    exp_cfg.runtime.mixed_precision_dtype = runtime['mixed_precision']
                if runtime.get('num_gpus'):
                    exp_cfg.runtime.num_gpus = runtime['num_gpus']
        except Exception:
            pass

        # Map data configuration
        try:
            if 'data' in gui_cfg:
                data = gui_cfg['data']
                
                # Training data
                if data.get('train_data'):
                    exp_cfg.task.train_data.input_path = data['train_data']
                if data.get('batch_size'):
                    exp_cfg.task.train_data.global_batch_size = int(data['batch_size'])
                if data.get('image_size') and isinstance(data['image_size'], list) and len(data['image_size']) >= 2:
                    exp_cfg.task.model.input_size = data['image_size'][:2]
                elif data.get('image_size'):
                    size = int(data['image_size']) if isinstance(data['image_size'], (int, str)) else 224
                    exp_cfg.task.model.input_size = [size, size]
                    
                # Validation data
                if data.get('val_data'):
                    exp_cfg.task.validation_data.input_path = data['val_data']
                    exp_cfg.task.validation_data.global_batch_size = int(data.get('batch_size', 32))
                
                # Number of classes
                if data.get('num_classes'):
                    exp_cfg.task.model.num_classes = int(data['num_classes'])
        except Exception as e:
            print(f"Error mapping data config: {e}")

        # Map model configuration
        try:
            if 'model' in gui_cfg:
                model = gui_cfg['model']
                
                if model.get('backbone_type'):
                    exp_cfg.task.model.backbone.type = model['backbone_type']
                if model.get('model_id'):
                    if model['backbone_type'] == 'resnet':
                        exp_cfg.task.model.backbone.resnet.model_id = int(model['model_id'])
                if model.get('dropout_rate') is not None:
                    exp_cfg.task.model.dropout_rate = float(model['dropout_rate'])
                if model.get('activation'):
                    exp_cfg.task.model.norm_activation.activation = model['activation']
        except Exception as e:
            print(f"Error mapping model config: {e}")

        # Map training configuration
        try:
            if 'training' in gui_cfg:
                training = gui_cfg['training']
                
                if training.get('epochs'):
                    exp_cfg.trainer.train_steps = int(training['epochs']) * 1000  # Approximate
                
                # Learning rate setup
                if training.get('initial_learning_rate'):
                    lr_type = training.get('learning_rate_type', 'exponential')
                    if lr_type == 'exponential':
                        exp_cfg.trainer.optimizer_config.learning_rate = {
                            'type': 'exponential',
                            'exponential': {
                                'initial_learning_rate': float(training['initial_learning_rate']),
                                'decay_steps': 10000,
                                'decay_rate': 0.96
                            }
                        }
                    elif lr_type == 'constant':
                        exp_cfg.trainer.optimizer_config.learning_rate = {
                            'type': 'constant',
                            'constant': {
                                'learning_rate': float(training['initial_learning_rate'])
                            }
                        }
                
                # Optimizer settings
                if training.get('momentum') and training.get('weight_decay'):
                    exp_cfg.trainer.optimizer_config.optimizer = {
                        'type': 'sgd',
                        'sgd': {
                            'momentum': float(training['momentum']),
                            'weight_decay': float(training['weight_decay'])
                        }
                    }
                
                # Loss configuration
                if training.get('label_smoothing') is not None:
                    exp_cfg.task.losses.label_smoothing = float(training['label_smoothing'])
        except Exception as e:
            print(f"Error mapping training config: {e}")

        # Map augmentation configuration
        try:
            if 'augmentation' in gui_cfg:
                aug = gui_cfg['augmentation']
                
                # Handle new augmentation structure
                if isinstance(aug, dict):
                    # Check for horizontal flip
                    hflip = aug.get('Horizontal Flip', {})
                    if hflip.get('enabled', False):
                        exp_cfg.task.train_data.aug_rand_hflip = True
                    else:
                        exp_cfg.task.train_data.aug_rand_hflip = False
                    
                    # Check for random cropping
                    crop = aug.get('Random Cropping', {})
                    if crop.get('enabled', False):
                        exp_cfg.task.train_data.aug_crop = True
                        # Set crop area range if available
                        min_area = crop.get('crop_area_min', 0.08)
                        max_area = crop.get('crop_area_max', 1.0)
                        exp_cfg.task.train_data.crop_area_range = [min_area, max_area]
                    else:
                        exp_cfg.task.train_data.aug_crop = False
                    
                    # Check for color jittering
                    color_jitter = aug.get('Color Jittering', {})
                    if color_jitter.get('enabled', False):
                        # Map to color jitter strength (simplified mapping)
                        hue_shift = color_jitter.get('hue_shift_limit', 20) / 50.0  # Normalize to 0-1
                        exp_cfg.task.train_data.color_jitter = hue_shift
                    else:
                        exp_cfg.task.train_data.color_jitter = 0.0
                
                # Fallback to legacy structure for backward compatibility
                else:
                    exp_cfg.task.train_data.aug_rand_hflip = bool(aug.get('aug_rand_hflip', True))
                    exp_cfg.task.train_data.aug_crop = bool(aug.get('aug_crop', True))
                    if aug.get('crop_area_range'):
                        exp_cfg.task.train_data.crop_area_range = aug['crop_area_range']
                    if aug.get('color_jitter') is not None:
                        exp_cfg.task.train_data.color_jitter = float(aug['color_jitter'])
                    if aug.get('randaug_magnitude') is not None:
                        exp_cfg.task.train_data.randaug_magnitude = int(aug['randaug_magnitude'])
        except Exception as e:
            print(f"Error mapping augmentation config: {e}")

        # Map training advanced settings
        try:
            if 'training_advanced' in gui_cfg:
                adv = gui_cfg['training_advanced']
                if adv.get('steps_per_loop'):
                    exp_cfg.trainer.steps_per_loop = int(adv['steps_per_loop'])
                if adv.get('checkpoint_interval'):
                    exp_cfg.trainer.checkpoint_interval = int(adv['checkpoint_interval'])
                if adv.get('validation_interval'):
                    exp_cfg.trainer.validation_interval = int(adv['validation_interval'])
                if adv.get('max_to_keep'):
                    exp_cfg.trainer.max_to_keep = int(adv['max_to_keep'])
        except Exception as e:
            print(f"Error mapping advanced training config: {e}")

        return exp_cfg
