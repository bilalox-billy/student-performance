"""
Test script to verify YAML configuration is loaded correctly
"""
from src.utils import read_yaml
from src.logger import logging

def test_yaml_config():
    """Test if YAML configuration loads correctly"""
    try:
        # Load configuration
        config = read_yaml('params.yaml')
        
        print("=" * 50)
        print("YAML Configuration Test")
        print("=" * 50)
        
        # Test model trainer config
        print("\n1. Model Trainer Configuration:")
        print(f"   - Model Path: {config['model_trainer']['trained_model_file_path']}")
        print(f"   - Min R2 Score: {config['model_trainer']['minimum_r2_score']}")
        
        # Test models config
        print("\n2. Models Configuration:")
        enabled_models = []
        disabled_models = []
        
        for model_name, model_config in config['models'].items():
            if model_config.get('enabled', True):
                enabled_models.append(model_name)
                param_count = sum(len(v) if isinstance(v, list) else 1 
                                for v in model_config['params'].values())
                print(f"   ✓ {model_name}: {len(model_config['params'])} parameters, "
                      f"{param_count} combinations")
            else:
                disabled_models.append(model_name)
                print(f"   ✗ {model_name}: DISABLED")
        
        print(f"\n   Total Enabled: {len(enabled_models)}")
        print(f"   Total Disabled: {len(disabled_models)}")
        
        # Test grid search config
        print("\n3. Grid Search Configuration:")
        gs_config = config['grid_search']
        print(f"   - CV Folds: {gs_config['cv']}")
        print(f"   - Scoring: {gs_config['scoring']}")
        print(f"   - Parallel Jobs: {gs_config['n_jobs']}")
        print(f"   - Verbose: {gs_config['verbose']}")
        
        # Test data config
        print("\n4. Data Configuration:")
        print(f"   - Test Size: {config['data_ingestion']['test_size']}")
        print(f"   - Random State: {config['data_ingestion']['random_state']}")
        print(f"   - Numerical Columns: {config['data_transformation']['numerical_columns']}")
        print(f"   - Categorical Columns: {len(config['data_transformation']['categorical_columns'])} columns")
        print(f"   - Target Column: {config['data_transformation']['target_column']}")
        
        print("\n" + "=" * 50)
        print("✓ Configuration loaded successfully!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error loading configuration: {e}")
        return False

if __name__ == "__main__":
    test_yaml_config()

