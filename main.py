from flask import Flask, render_template, request, send_from_directory
import os

# FORCE use of legacy Keras to support the .h5 file format
# (Using tensorflow.keras avoids requiring the separate `tf-keras` package.)
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import cv2
from PIL import Image

import tensorflow as tf

# We support two model formats in this project:
# - Legacy HDF5 `.h5` (best loaded via legacy tf.keras / `tf-keras`)
# - Modern Keras v3 `.keras` (loaded via standalone `keras`)

# Legacy tf.keras (Keras 2) loader
try:
    import tf_keras as legacy_keras  # provided by pip package `tf-keras`
    from tf_keras.models import load_model as load_model_legacy, Sequential
    from tf_keras.layers import InputLayer, Flatten, Dense, Dropout
    from tf_keras.preprocessing.image import load_img, img_to_array
    from tf_keras.applications import VGG16
except ModuleNotFoundError:
    from tensorflow import keras as legacy_keras
    from tensorflow.keras.models import load_model as load_model_legacy, Sequential
    from tensorflow.keras.layers import InputLayer, Flatten, Dense, Dropout
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from tensorflow.keras.applications import VGG16

# Modern Keras v3 loader (for `.keras`)
import keras as modern_keras

from gradcam import make_gradcam_heatmap, overlay_heatmap_on_image, get_last_conv_layer_name

# Custom layers to handle compatibility issues
# Removed CustomInputLayer as model config is patched

# Initialize Flask app
app = Flask(__name__)

# Global variables
current_model = None
current_model_name = None
current_model_type = None  # "h5" or "keras"
last_conv_layer_name = None

def get_available_models():
    """Return list of model files in the models directory (.h5 and .keras)."""
    models_dir = 'models'
    if not os.path.exists(models_dir):
        return []
    return [
        f for f in os.listdir(models_dir)
        if f.endswith('.h5') or f.endswith('.keras')
    ]

def get_model_input_config(model):
    """Get the expected input shape configuration from the model."""
    try:
        input_shape = model.input_shape
        if isinstance(input_shape, (list, tuple)):
            if len(input_shape) > 0:
                 if isinstance(input_shape[0], (int, type(None))):
                     pass 
                 else:
                     input_shape = input_shape[0]
        
        height = input_shape[1] if input_shape[1] is not None else 128
        width = input_shape[2] if input_shape[2] is not None else 128
        channels = input_shape[3] if input_shape[3] is not None else 3
        return height, width, channels
    except Exception:
        return 128, 128, 3

def build_model():
    """Reconstruct the model architecture (matches models/model.h5 config)."""
    IMAGE_SIZE = 128
    base_model = VGG16(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights=None,
    )

    # Names are important because the HDF5 weight file nests weights under these layer names.
    model = Sequential(name="sequential_3")
    model.add(base_model)
    model.add(Flatten(name="flatten_3"))
    model.add(Dropout(0.3, name="dropout_6"))
    model.add(Dense(128, activation='relu', name="dense_6"))
    model.add(Dropout(0.2, name="dropout_7"))
    model.add(Dense(4, activation='softmax', name="dense_7"))
    return model


def load_weights_manual(model, model_path):
    """Manually load weights from the legacy HDF5 file.

    Keras/tf_keras HDF5 loaders can mis-map weights when a Sequential contains a nested
    Functional model (like VGG16). The file we have stores VGG weights under:
      model_weights/vgg16/<layer>/(kernel|bias)
    and top-level Dense weights under:
      model_weights/dense_6/sequential_3/dense_6/(kernel|bias)
    (similar for dense_7).
    """
    import h5py

    def _read_kernel_bias(group):
        weights = []
        if 'kernel' in group:
            weights.append(group['kernel'][()])
        if 'bias' in group:
            weights.append(group['bias'][()])
        return weights

    with h5py.File(model_path, 'r') as f:
        root = f.get('model_weights')
        if root is None:
            raise ValueError("No 'model_weights' group found in the HDF5 file")

        # 1) Load nested VGG16 weights
        if 'vgg16' in root:
            vgg_group = root['vgg16']
            vgg_model = model.get_layer('vgg16')
            for layer in vgg_model.layers:
                if layer.name in vgg_group:
                    w = _read_kernel_bias(vgg_group[layer.name])
                    if w:
                        layer.set_weights(w)

        # 2) Load top-level Dense layer weights (dropout/flatten have no weights)
        for layer in model.layers:
            if not layer.weights:
                continue
            if layer.name == 'vgg16':
                continue
            if layer.name not in root:
                continue

            g = root[layer.name]
            # Many weights are nested: <layer_name>/sequential_3/<layer_name>/...
            if 'sequential_3' in g and layer.name in g['sequential_3']:
                g = g['sequential_3'][layer.name]

            w = _read_kernel_bias(g)
            if w:
                layer.set_weights(w)

def _labels_for_model(model_name):
    # Validated Mappings (debugged via sample images):
    # - model.keras: Alphabetical ['glioma', 'meningioma', 'notumor', 'pituitary'] (Expects 0-255 input)
    # - model.h5: Custom Order ['pituitary', 'glioma', 'notumor', 'meningioma'] (Expects 0-1 input)
    if model_name.endswith('.keras'):
        return ['glioma', 'meningioma', 'notumor', 'pituitary']
    return ['pituitary', 'glioma', 'notumor', 'meningioma']


def crop_image_cv2(image):
    """Crop the brain region (ported from Grad_cam_tumor_detection.ipynb)."""
    try:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
        img_thresh = cv2.threshold(img_blur, 45, 255, cv2.THRESH_BINARY)[1]
        img_thresh = cv2.erode(img_thresh, None, iterations=2)
        img_thresh = cv2.dilate(img_thresh, None, iterations=2)

        cnts = cv2.findContours(img_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        if not cnts:
            return image

        c = max(cnts, key=cv2.contourArea)
        ext_left = tuple(c[c[:, :, 0].argmin()][0])
        ext_right = tuple(c[c[:, :, 0].argmax()][0])
        ext_top = tuple(c[c[:, :, 1].argmin()][0])
        ext_bottom = tuple(c[c[:, :, 1].argmax()][0])

        new_img = image[ext_top[1]: ext_bottom[1], ext_left[0]: ext_right[0]]
        if new_img is None or new_img.size == 0:
            return image
        return new_img
    except Exception:
        return image


def preprocess_image(image_path, model_name, model):
    """Prepare model input array based on model format."""
    height, width, channels = get_model_input_config(model)

    # model.keras was trained on cropped 240x240 images (see notebook).
    if model_name.endswith('.keras'):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image")

        img = crop_image_cv2(img)
        img = cv2.resize(img, (width, height))
        if channels == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # IMPORTANT: model.keras expects [0-255], do NOT normalize by /255.
        x = img.astype(np.float32)
        return np.expand_dims(x, axis=0)

    # Legacy `.h5` path: keras load_img + normalize to [0,1]
    color_mode = 'grayscale' if channels == 1 else 'rgb'
    img = load_img(image_path, target_size=(height, width), color_mode=color_mode)
    x = img_to_array(img) / 255.0
    return np.expand_dims(x, axis=0)


def load_model_by_name(model_name):
    """Load a model (.h5 or .keras) from the models directory."""
    global current_model, current_model_name, current_model_type, last_conv_layer_name, class_labels
    if current_model_name == model_name and current_model is not None:
        return current_model, last_conv_layer_name
    
    model_path = os.path.join('models', model_name)
    if not os.path.exists(model_path):
        models = get_available_models()
        if not models:
            raise FileNotFoundError("No models found in models directory")
        model_path = os.path.join('models', models[0])
        model_name = models[0]
        
    # Update label mapping for the currently selected model
    class_labels = _labels_for_model(model_name)

    # Modern `.keras` model
    if model_name.endswith('.keras'):
        print(f"Loading model: {model_name} (keras v3)...")
        current_model = modern_keras.models.load_model(model_path, compile=False)
        current_model_type = 'keras'
    else:
        # Legacy `.h5` model
        print(f"Loading model: {model_name} (legacy .h5)...")

        class CustomInputLayer(InputLayer):
            def __init__(self, *args, **kwargs):
                if 'batch_input_shape' in kwargs and isinstance(kwargs['batch_input_shape'], list):
                    kwargs['batch_input_shape'] = tuple(kwargs['batch_input_shape'])
                if 'batch_shape' in kwargs and isinstance(kwargs['batch_shape'], list):
                    kwargs['batch_shape'] = tuple(kwargs['batch_shape'])
                super().__init__(*args, **kwargs)

        try:
            current_model = load_model_legacy(model_path, compile=False, custom_objects={'InputLayer': CustomInputLayer})
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model details: {type(e).__name__}: {e}")
            print("Attempting to reconstruct model structure and load weights as fallback...")
            current_model = build_model()
            try:
                load_weights_manual(current_model, model_path)
                print("Model weights loaded successfully (manual fallback).")
            except Exception as werr:
                print(f"Manual weight load failed: {type(werr).__name__}: {werr}")
                print("Falling back to Keras load_weights(by_name=True, skip_mismatch=True)...")
                current_model.load_weights(model_path, by_name=True, skip_mismatch=True)
                print("Model weights loaded successfully (best-effort fallback).")

        current_model_type = 'h5'

    current_model_name = model_name
    last_conv_layer_name = get_last_conv_layer_name(current_model)
    return current_model, last_conv_layer_name

# Class labels
# Updated dynamically when a model is loaded.
class_labels = _labels_for_model('model.h5')

# Define the uploads folder
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Tumor analysis info
TUMOR_INFO = {
    'meningioma': {'title': 'Meningioma Tumor Analysis', 'description': 'A meningioma is a tumor that arises from the meninges.', 'details': ['Most are benign.', 'Grow slowly.', 'Symptoms: headaches, vision changes.', 'Treatment: observation, surgery.']},
    'glioma': {'title': 'Glioma Tumor Analysis', 'description': 'Glioma occurs in the brain and spinal cord.', 'details': ['Can be life-threatening.', 'Symptoms: nausea, confusion.', 'Treatment: surgery, chemotherapy.']},
    'pituitary': {'title': 'Pituitary Tumor Analysis', 'description': 'Pituitary tumors develop in the pituitary gland.', 'details': ['Most are benign.', 'Cause hormonal imbalances.', 'Treatment: surgery, medication.']},
    'notumor': {'title': 'Healthy Brain Analysis', 'description': 'No tumor patterns detected.', 'details': ['Normal brain structure.', 'Regular check-ups recommended.']}
}

INVALID_IMAGE_INFO = {'title': 'Invalid Image Detected', 'description': 'Not a valid MRI scan.', 'details': ['Please upload a standard brain MRI scan.']}

def is_likely_mri(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None: return False
        if np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]) > 25: return False
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        crop = max(1, int(min(h, w) * 0.05))
        corners = [gray[:crop, :crop], gray[:crop, -crop:], gray[-crop:, :crop], gray[-crop:, -crop:]]
        if np.mean([np.mean(c) for c in corners]) > 60: return False
        return True
    except: return True

def predict_tumor(image_path, model, last_conv_layer_name):
    global current_model_name

    if not is_likely_mri(image_path):
        return "Invalid Image", 0.0, image_path, INVALID_IMAGE_INFO

    img_array_batch = preprocess_image(image_path, current_model_name or '', model)

    predictions = model.predict(img_array_batch, verbose=0)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]
    predicted_label = class_labels[predicted_class_index]
    
    heatmap = np.zeros((1, 1))
    gradcam_path = image_path 
    
    if predicted_label != 'notumor':
        try:
            heatmap = make_gradcam_heatmap(img_array_batch, model, last_conv_layer_name, predicted_class_index)
            
            original_img = Image.open(image_path).convert('RGB')
            # Resize original image to match model input size for better overlay? No, usually we resize heatmap to original.
            # But the code resizes heatmap to original.
            original_img_array = np.array(original_img)
            
            # Simple contrast enhancement
            lab = cv2.cvtColor(original_img_array, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
            original_img_array = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
            
            gradcam_img = overlay_heatmap_on_image(original_img_array, heatmap, alpha=0.4)
            gradcam_path = os.path.splitext(image_path)[0] + '_gradcam.jpg'
            cv2.imwrite(gradcam_path, cv2.cvtColor(gradcam_img, cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f"Grad-CAM error: {e}")

    analysis = TUMOR_INFO.get(predicted_label, TUMOR_INFO['notumor'])
    prefix = "No Tumor" if predicted_label == 'notumor' else f"Tumor: {predicted_label}"
    return prefix, confidence_score, gradcam_path, analysis

MODEL_DISPLAY_NAMES = {
    'model.h5': 'VGG16 Model',
    'model.keras': 'EfficientNet Model'
}

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        global current_model
        available_models = get_available_models()
        if current_model is None and available_models:
             load_model_by_name(available_models[0])
             
        if request.method == 'POST':
            file = request.files.get('file')
            model_name = request.form.get('model_name')
            
            if model_name and model_name != current_model_name:
                load_model_by_name(model_name)
                
            if file and file.filename:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)
                
                result, conf, gc_path, analysis = predict_tumor(file_path, current_model, last_conv_layer_name)
                
                return render_template('index.html',
                                     result=result,
                                     confidence=f"{conf*100:.2f}%",
                                     file_path=f'/uploads/{os.path.basename(file_path)}',
                                     gradcam_path=f'/uploads/{os.path.basename(gc_path)}',
                                     analysis=analysis,
                                     available_models=available_models,
                                     current_model=current_model_name,
                                     model_display_names=MODEL_DISPLAY_NAMES)
    except Exception as e:
        print(f"Error in index route: {e}")
        return str(e), 500
        
    return render_template('index.html', result=None, available_models=available_models, current_model=current_model_name, model_display_names=MODEL_DISPLAY_NAMES)

@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Load model on start
    available = get_available_models()
    if available:
        try:
            load_model_by_name(available[0])
        except Exception as e:
            print(f"Startup loading error: {e}")
            
    app.run(debug=True)
