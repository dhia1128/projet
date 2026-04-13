"""
Model caching utility using joblib
Saves/loads trained models and their preprocessing objects (scalers, etc.)
"""
import joblib
from pathlib import Path
import os
from typing import Any, Tuple

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def save_model(model: Any, model_name: str, **metadata) -> str:
    """
    Save a model and optional metadata using joblib
    
    Args:
        model: Model/object to save
        model_name: Name identifier for the model
        **metadata: Additional objects to save (scalers, parameters, etc.)
    
    Returns:
        Path where model was saved
    """
    model_path = MODELS_DIR / f"{model_name}.joblib"
    
    # Bundle model with metadata
    package = {"model": model, **metadata}
    
    joblib.dump(package, model_path, compress=3)  # compress=3 for better compression
    print(f"✓ Model saved: {model_path}")
    return str(model_path)


def load_model(model_name: str) -> Tuple[Any, dict]:
    """
    Load a model and its metadata from joblib cache
    
    Args:
        model_name: Name identifier for the model
    
    Returns:
        Tuple of (model, metadata_dict)
    """
    model_path = MODELS_DIR / f"{model_name}.joblib"
    
    if not model_path.exists():
        return None, {}
    
    package = joblib.load(model_path)
    model = package.pop("model")
    
    print(f"✓ Model loaded from cache: {model_path}")
    return model, package


def model_exists(model_name: str) -> bool:
    """Check if model cache exists"""
    model_path = MODELS_DIR / f"{model_name}.joblib"
    return model_path.exists()


def clear_model_cache(model_name: str) -> bool:
    """Delete cached model"""
    model_path = MODELS_DIR / f"{model_name}.joblib"
    if model_path.exists():
        model_path.unlink()
        print(f"✓ Cache cleared: {model_path}")
        return True
    return False
