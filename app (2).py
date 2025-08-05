import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Rebuild your original CNN architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')  # 10 EuroSAT classes
])

# Load the weights
model.load_weights("land_classifier_model.h5")

# Class labels
class_names = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]

# Prediction function
def classify_image(image):
    try:
        image = image.convert("RGB")
        image = image.resize((64, 64))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = round(np.max(prediction) * 100, 2)

        return f"Prediction: {predicted_class} ({confidence}%)"
    except Exception as e:
        return f"❌ Error during prediction: {str(e)}"

# Gradio UI
interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="Upload JPG/PNG only"),
    outputs="text",
    title="🌍 Satellite Land Use Classifier",
    description="Upload a satellite image (.jpg/.png only) to classify land type using a CNN trained on EuroSAT data."
)

interface.launch()