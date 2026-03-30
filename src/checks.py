import sys

# --- Check Python Version ---
required_python = (3, 13)  # Change to your minimum required version
current_python = sys.version_info[:2]

print(f"Current Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
if current_python < required_python:
    print(f"⚠️ Python {required_python[0]}.{required_python[1]} or higher is required.")
else:
    print("✅ Python version is OK.")

# --- Check TensorFlow ---
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    # Optional: test if GPU is available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ TensorFlow GPU detected: {[gpu.name for gpu in gpus]}")
    else:
        print("⚠️ No GPU detected, using CPU.")
except ImportError:
    print("❌ TensorFlow is not installed. Install it with `pip install tensorflow`.")
except Exception as e:
    print(f"❌ TensorFlow detected but error occurred: {e}")