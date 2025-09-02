new design principles

remove all gui designs, only keep cli interface
✅ COMPLETED: moved the callbacks, data augmentation and k-fold-crossvalidation configuration to basic, then removed all advanced configurations

the path to the custom functions should be relative to the yaml file
all custom functions should be placed in a directory named `src`

# generate python scripts: train.py, evaluation.py prediction.py, deploy.py according to the yaml configuration and save to the directory same with yaml file.

don't provide presets models for section any more, in which can make the logic more complex

In the YAML configuration files, all build-in configurations will be commented out by default. The user can uncomment and modify these configurations as needed.

In the CLI interface, the create command will create a project that with all custom_function settings, which makes the user easier for customizing their own project.

## Configuration Structure Improvements

The configuration has been simplified to a single "basic" level structure:

**Before (Complex):**
- basic/ (essential settings)
- advanced/ (expert settings with callbacks, augmentation, cross-validation)

**After (Simplified):**
- configuration/
  - data/ (includes augmentation)
  - model/ (includes callbacks)
  - training/ (includes cross-validation)
  - runtime/

### Key Changes Made:
1. **Callbacks**: `advanced > callbacks` → `basic > model > callbacks`
2. **Data Augmentation**: `advanced > augmentation` → `basic > data > augmentation`
3. **K-fold Cross-validation**: `advanced > cross_validation` → `basic > training > cross_validation`
4. **Advanced Section**: Completely removed for simplification

This makes the configuration more intuitive and accessible to all users.ciples

remove all gui designs, only keep cli interface
x move the callbacks, data augmentation and k-fold-crossvalidation configuration to basic, then remove all advanced configurations

the path to the custom functions should be relative to the yaml file
all custom functions should be placed in a directory named `src`

# generate python scripts: train.py, evaluation.py prediction.py, deploy.py according to the yaml configuration and save to the directory same with yaml file.

don't provide presets models for section any more, in which can make the logic more complex

In the YAML configuration files, all build-in configurations will be commented out by default. The user can uncomment and modify these configurations as needed.

In the CLI interface, the create command will create a project that with all custom_function settings, which makes the user easier for customizing their own project.

