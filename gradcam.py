import numpy as np
import tensorflow as tf

# Support both legacy `tf_keras` models (Keras 2) and modern `keras` models (Keras 3).
try:
    import tf_keras as tfk
except ModuleNotFoundError:
    tfk = None

try:
    import keras as k3
except ModuleNotFoundError:
    from tensorflow import keras as k3


def _pick_keras(model):
    """Pick a Keras module compatible with the given model instance."""
    mod = type(model).__module__
    if tfk is not None and mod.startswith('tf_keras'):
        return tfk
    return k3

import cv2

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate a Grad-CAM heatmap.

    Notes:
    - When the conv layer is inside a nested Functional model (e.g. VGG16 inside a
      Sequential), using `conv_layer.output` directly can produce a "Graph disconnected"
      error because that tensor belongs to the nested model's original graph.
    - To avoid that, we rebuild a small functional graph that reuses the same layers
      and produces both (conv_outputs, predictions) from the same forward pass.
    """

    K = _pick_keras(model)

    model_in = model.inputs
    if isinstance(model_in, (list, tuple)) and len(model_in) == 1:
        model_in = model_in[0]

    if '/' in last_conv_layer_name:
        parent_name, child_name = last_conv_layer_name.split('/')
        base = model.get_layer(parent_name)

        base_in = base.inputs
        if isinstance(base_in, (list, tuple)) and len(base_in) == 1:
            base_in = base_in[0]

        # Multi-output version of the nested model: [target conv activation, base output]
        base_multi = K.Model(
            inputs=base_in,
            outputs=[base.get_layer(child_name).output, base.output],
        )

        # Rebuild the classifier head using the existing layers after the base model.
        x_in = model_in
        conv_outputs, x = base_multi(x_in)
        y = x
        for layer in model.layers[1:]:
            y = layer(y)

        grad_model = K.Model(x_in, [conv_outputs, y])
    else:
        conv_layer = model.get_layer(last_conv_layer_name)
        grad_model = K.Model(model_in, [conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        # If the model has multiple inputs, Keras expects a list/tuple.
        call_arg = img_array
        if isinstance(grad_model.inputs, (list, tuple)) and len(grad_model.inputs) > 1:
            call_arg = [img_array]

        conv_outputs, predictions = grad_model(call_arg)

        # Some Keras models expose single outputs as a list of length 1.
        if isinstance(conv_outputs, (list, tuple)) and len(conv_outputs) == 1:
            conv_outputs = conv_outputs[0]
        if isinstance(predictions, (list, tuple)) and len(predictions) == 1:
            predictions = predictions[0]

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    
    # This is a vector where each entry is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Multiply each channel in the feature map array by its importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


def overlay_heatmap_on_image(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """Overlay the heatmap on the original image."""
    # Convert PIL image to numpy array if needed
    if hasattr(img, 'size'):  # PIL Image
        img = np.array(img)
    
    # Resize heatmap to match image dimensions
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    
    # Convert BGR to RGB (OpenCV uses BGR by default)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Blend the heatmap with the original image
    superimposed_img = cv2.addWeighted(heatmap, alpha, img, 1.0, 0)
    
    return superimposed_img


def get_last_conv_layer_name(model):
    """Find the last convolutional layer in the model.

    Supports both Conv2D and DepthwiseConv2D.
    """
    K = _pick_keras(model)
    conv_types = (K.layers.Conv2D, K.layers.DepthwiseConv2D)

    # First check the main model layers in reverse
    for layer in reversed(model.layers):
        if isinstance(layer, conv_types):
            return layer.name

        # If it's a functional or sequential model, search within it
        if hasattr(layer, 'layers'):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, conv_types):
                    return f"{layer.name}/{sub_layer.name}"

    raise ValueError("No convolution layer found in the model")
