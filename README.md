new design principles

remove all gui designs, only keep cli interface
move the callbacks, data augmentation and k-fold-crossvalidation configuration to basic, then remove all advanced configurations
change the configuration saving file format to YAML

don't provide presets models for section any more, in which can make the logic more complex

In the YAML configuration files, all build-in configurations will be commented out by default. The user can uncomment and modify these configurations as needed.

In the CLI interface, the create command will create a project that with all custom_function settings, which makes the user easier for customizing their own project.

