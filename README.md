🌿✨ Plant Disease Recognition Using Transfer Learning
Deep Learning–Powered Early Detection for Healthier Crops

Welcome to the Plant Disease Recognition project — a high-performance deep learning pipeline built to identify plant leaf diseases using state-of-the-art transfer learning. This repository contains a clean, modular, and production-ready Jupyter Notebook that leverages MobileNetV2, advanced image preprocessing, and evaluation visualizations.

This project is perfect for:

🌱 Agricultural AI research

📊 Machine learning portfolio building

👨‍🏫 Deep learning students exploring transfer learning

🌍 Real-world plant health monitoring applications

🚀 Project Highlights

⚡ End-to-End Deep Learning Pipeline — from data loading to prediction

🌐 MobileNetV2 Feature Extraction for state-of-the-art accuracy

🖼️ Image Preprocessing & Augmentation built with TensorFlow

📉 Training & Validation Curve Visualization

🎯 Multi-Class Disease Classification using softmax output

🧪 Prediction on Custom Images

📦 Minimal Install & Easy Reproducibility

📁 Dataset Structure

The notebook works with a dataset structured like:

Train/
    ├── Class_A/
    ├── Class_B/
    ├── Class_C/
Validation/
    ├── Class_A/
    ├── Class_B/
    ├── Class_C/


Dataset paths used in the notebook:

D:\ML DATASETS\plant disease recognition\Train\Train
D:\ML DATASETS\plant disease recognition\validation\validation


Update these paths based on your system.

🔧 Technologies & Libraries Used
🔹 📦 Library Installation
!pip install tensorflow opencv-python matplotlib scikit-learn

🔹 📚 Core Imports

This project utilizes:

TensorFlow / Keras

NumPy

Matplotlib

OpenCV

scikit-learn

🔹 🗂️ Dataset Loading & Preprocessing

Image preprocessing includes:

Resizing to 224x224

Normalization

Batch loading

Auto-labeled directory reading

Using TensorFlow’s ImageDataGenerator.

🔹 🧠 Transfer Learning with MobileNetV2

Key model definition:

base = MobileNetV2(weights='imagenet', include_top=False,
                   input_shape=IMG_SIZE + (3,))
base.trainable = False


A fully custom classification head is then added.

🔹 🏋️ Model Training

Loss: Categorical Crossentropy

Optimizer: Adam

Epochs: 10

Training executed via:

history = model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)

🔹 📈 Performance Visualization

Generates:

📊 Accuracy plots

📉 Loss plots

All exported as .png files.

🔹 🧪 Model Evaluation

Predictions produced using:

Y_pred = model.predict(val_generator)


Includes:

Softmax probabilities

Class mapping

Label decoding

🔹 🔍 Predictions on Custom Images

Image prediction pipeline:

img = image.load_img(path, target_size=IMG_SIZE)


Then preprocessed → fed to classifier → outputs label.

💡 How to Use This Repository
1️⃣ Clone the Repo
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

2️⃣ Install Requirements
pip install -r requirements.txt

3️⃣ Open the Notebook
jupyter notebook

4️⃣ Update Dataset Paths

Modify:

train_dir = r"..."
val_dir   = r"..."

5️⃣ Run All Cells
🌟 Output Examples

✔️ Predicted class labels

✔️ Accuracy/loss curves

✔️ Saved prediction images

✔️ Validation results

📌 Future Improvements

🔄 Fine-tune MobileNet deeper layers

📊 Add confusion matrix

🌐 Deploy via FastAPI / Gradio

🧪 Add separate test dataset

🤝 Contributing

PRs, issues, and suggestions are welcome!

⭐ Support the Project

If you like this repository, please ⭐ star it on GitHub!
