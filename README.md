---
title: Satellite Landuse Classifier
emoji: 🌍
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.39.0
app_file: app.py
pinned: false
license: mit
---

# 🌍 Satellite Land Use Classifier

This project is a deep learning application that classifies satellite images into land use categories (e.g., Forest, Residential, Industrial, etc.) using a Convolutional Neural Network (CNN) trained on the EuroSAT dataset.

## 📌 Features

- Upload a `.jpg` or `.png` satellite image
- Predicts land use type such as Forest, Residential, Highway, Industrial, etc.
- Built with TensorFlow + Gradio
- Deployed on Hugging Face Spaces

## 🧠 Model Details

- **Dataset:** EuroSAT (RGB)
- **Model Type:** CNN with Conv2D + MaxPooling layers
- **Classes:** Forest, Industrial, Residential, Pasture, River, Highway, etc.
- **Framework:** TensorFlow/Keras


👉 Try it now: [Live on Hugging Face](https://huggingface.co/spaces/DanaAlzahid/satellite-landuse-classifier)


## 📷 Example Predictions

| Image | Prediction |
|-------|------------|
| Forest | ✅ Correct |
| River  | ❌ Wrong: Predicted as Highway |
| Industrial | ✅ Correct |

## 👩‍💻 How to Use

1. Upload a satellite image
2. Click Submit
3. View predicted land use class and probability

## 📜 License

MIT — free to use and modify.

---

Let me know if you want to **auto-generate a download for this `README.md`** or publish it to GitHub too.

