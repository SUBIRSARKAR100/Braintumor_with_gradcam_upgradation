import os
import cv2
import numpy as np
import tensorflow as tf
import keras as modern_keras
import sys
import h5py

sys.stdout.reconfigure(encoding='utf-8')

# Force legacy keras for h5
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tf_keras as legacy_keras
from tf_keras.models import load_model as load_model_legacy, Sequential
from tf_keras.layers import InputLayer, Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from tf_keras.applications import VGG16

def build_model():
    # Reconstruct the model architecture (matches models/model.h5 config)
    IMAGE_SIZE = 128
    base_model = VGG16(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights=None,
    )
    model = Sequential(name="sequential_3")
    model.add(base_model)
    model.add(Flatten(name="flatten_3"))
    model.add(Dropout(0.3, name="dropout_6"))
    model.add(Dense(128, activation='relu', name="dense_6"))
    model.add(Dropout(0.2, name="dropout_7"))
    model.add(Dense(4, activation='softmax', name="dense_7"))
    return model

def load_weights_manual(model, model_path):
    def _read_kernel_bias(group):
        weights = []
        if 'kernel' in group: weights.append(group['kernel'][()])
        if 'bias' in group: weights.append(group['bias'][()])
        return weights

    with h5py.File(model_path, 'r') as f:
        root = f.get('model_weights')
        if root is None: return
        
        # Load VGG
        if 'vgg16' in root:
            vgg_group = root['vgg16']
            vgg_model = model.get_layer('vgg16')
            for layer in vgg_model.layers:
                if layer.name in vgg_group:
                    w = _read_kernel_bias(vgg_group[layer.name])
                    if w: layer.set_weights(w)

        # Load Dense
        for layer in model.layers:
            if not layer.weights or layer.name == 'vgg16': continue
            if layer.name not in root: continue
            g = root[layer.name]
            if 'sequential_3' in g and layer.name in g['sequential_3']:
                g = g['sequential_3'][layer.name]
            w = _read_kernel_bias(g)
            if w: layer.set_weights(w)

def crop_image_cv2(image):
    try:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
        img_thresh = cv2.threshold(img_blur, 45, 255, cv2.THRESH_BINARY)[1]
        img_thresh = cv2.erode(img_thresh, None, iterations=2)
        img_thresh = cv2.dilate(img_thresh, None, iterations=2)
        cnts = cv2.findContours(img_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        if not cnts: return image
        c = max(cnts, key=cv2.contourArea)
        ext_left = tuple(c[c[:, :, 0].argmin()][0])
        ext_right = tuple(c[c[:, :, 0].argmax()][0])
        ext_top = tuple(c[c[:, :, 1].argmin()][0])
        ext_bottom = tuple(c[c[:, :, 1].argmax()][0])
        new_img = image[ext_top[1]: ext_bottom[1], ext_left[0]: ext_right[0]]
        return new_img if new_img.size != 0 else image
    except: return image

def run_debug():
    images = {
        'glioma': 'sample_MRI_Images/Te-gl_0015.jpg',
        'meningioma': 'sample_MRI_Images/Te-meTr_0001.jpg',
        'notumor': 'sample_MRI_Images/Te-noTr_0004.jpg',
        'pituitary': 'sample_MRI_Images/Te-piTr_0003.jpg'
    }

    with open('results.txt', 'w', encoding='utf-8') as f:
        f.write("Loading Models...\n")
        
        # Keras V3
        try:
            model_keras = modern_keras.models.load_model('models/model.keras', compile=False)
            f.write("Loaded model.keras\n")
        except Exception as e:
            f.write(f"Failed to load model.keras: {e}\n")
            model_keras = None

        # H5 Legacy
        try:
            class CustomInputLayer(InputLayer):
                 def __init__(self, *args, **kwargs):
                     if 'batch_input_shape' in kwargs and isinstance(kwargs['batch_input_shape'], list):
                         kwargs['batch_input_shape'] = tuple(kwargs['batch_input_shape'])
                     if 'batch_shape' in kwargs and isinstance(kwargs['batch_shape'], list):
                         kwargs['batch_shape'] = tuple(kwargs['batch_shape'])
                     super().__init__(*args, **kwargs)
            model_h5 = load_model_legacy('models/model.h5', compile=False, custom_objects={'InputLayer': CustomInputLayer})
            f.write("Loaded model.h5 (direct)\n")
        except Exception:
            f.write("Fallback load model.h5...\n")
            model_h5 = build_model()
            load_weights_manual(model_h5, 'models/model.h5')
            f.write("Loaded model.h5 (manual)\n")

        test_models = [('model.keras', model_keras), ('model.h5', model_h5)]

        for model_name, model in test_models:
            if model is None: continue
            f.write(f"\n--- Testing {model_name} ---\n")
            
            try:
                 s = model.input_shape
                 if isinstance(s, list): s=s[0]
                 h, w = s[1], s[2]
            except: h, w = 128, 128
            
            for true_label, img_path in images.items():
                if not os.path.exists(img_path): continue
                
                img_bgr = cv2.imread(img_path)
                img_cropped = crop_image_cv2(img_bgr)
                img_resized = cv2.resize(img_cropped, (w, h))
                img_rgb_base = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                
                # 0-1
                x1 = np.expand_dims(img_rgb_base.astype(np.float32) / 255.0, axis=0)
                p1 = model.predict(x1, verbose=0)[0]
                
                # 0-255
                x2 = np.expand_dims(img_rgb_base.astype(np.float32), axis=0)
                p2 = model.predict(x2, verbose=0)[0]

                f.write(f"Img: {true_label}\n")
                f.write(f"  [0-1]   Idx: {np.argmax(p1)} Conf: {np.max(p1):.4f} Vec: {p1}\n")
                f.write(f"  [0-255] Idx: {np.argmax(p2)} Conf: {np.max(p2):.4f} Vec: {p2}\n")

if __name__ == "__main__":
    run_debug()
