#!/usr/bin/env python3
"""
Test script to validate the residual vs data training configuration and imports.
"""

import yaml
import os
import sys

def test_config_loading():
    """Test that configuration files can be loaded correctly."""
    config_files = [
        'configs/residual_vs_data_training.yml',
        'configs/residual_vs_data_training_data_mode.yml'
    ]

    for config_file in config_files:
        if not os.path.exists(config_file):
            print(f"❌ Config file not found: {config_file}")
            continue

        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            # Validate required keys
            required_keys = ['seed', 'data', 'model', 'training', 'evaluation']
            missing_keys = [key for key in required_keys if key not in config]

            if missing_keys:
                print(f"❌ {config_file}: Missing keys: {missing_keys}")
            else:
                print(f"✅ {config_file}: Valid configuration")
                print(f"   Mode: {config['training']['mode']}")
                print(f"   N_u: {config['data']['N_u']}, N_f: {config['data']['N_f']}")
                print(f"   Epochs: {config['training']['epochs']}")

        except Exception as e:
            print(f"❌ {config_file}: Error loading - {e}")


def test_imports():
    """Test that all required modules can be imported."""
    imports_to_test = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('scipy.io', 'SciPy'),
        ('matplotlib.pyplot', 'Matplotlib'),
        ('yaml', 'PyYAML'),
    ]

    print("\n--- Testing Standard Library Imports ---")
    for module_name, display_name in imports_to_test:
        try:
            __import__(module_name)
            print(f"✅ {display_name}: Available")
        except ImportError as e:
            print(f"❌ {display_name}: Not available - {e}")

    # Test custom package imports
    print("\n--- Testing Custom Package Imports ---")
    custom_imports = [
        'constrained_nn_eq_discovery.dhpm',
        'constrained_nn_eq_discovery.training',
        'constrained_nn_eq_discovery.con_opt',
        'constrained_nn_eq_discovery.numerics',
        'constrained_nn_eq_discovery.interpolate',
        'constrained_nn_eq_discovery.utils',
    ]

    for module_name in custom_imports:
        try:
            __import__(module_name)
            print(f"✅ {module_name}: Available")
        except ImportError as e:
            print(f"❌ {module_name}: Not available - {e}")


def test_data_file():
    """Test that the data file exists."""
    data_file = 'data/burgers_sine.mat'
    if os.path.exists(data_file):
        print(f"✅ Data file found: {data_file}")
        file_size = os.path.getsize(data_file) / (1024 * 1024)  # MB
        print(f"   File size: {file_size:.2f} MB")
    else:
        print(f"❌ Data file not found: {data_file}")


def test_output_directories():
    """Test that output directories exist or can be created."""
    output_dirs = [
        'models/residual_vs_data',
        'figs/residual_vs_data'
    ]

    print("\n--- Testing Output Directories ---")
    for dir_path in output_dirs:
        try:
            os.makedirs(dir_path, exist_ok=True)
            if os.path.exists(dir_path):
                print(f"✅ Directory ready: {dir_path}")
            else:
                print(f"❌ Could not create directory: {dir_path}")
        except Exception as e:
            print(f"❌ Error with directory {dir_path}: {e}")


def main():
    """Run all tests."""
    print("=== Residual vs Data Training - Configuration Test ===\n")

    print("--- Testing Configuration Files ---")
    test_config_loading()

    test_imports()

    print("\n--- Testing Data Files ---")
    test_data_file()

    test_output_directories()

    print("\n=== Test Summary ===")
    print("If all tests pass (✅), you can run the training script with:")
    print("python scripts/residual_vs_data_training.py --config configs/residual_vs_data_training.yml")


if __name__ == "__main__":
    main()
