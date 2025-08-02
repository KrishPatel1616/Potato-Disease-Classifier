🥔 Potato Disease Classifier
A basic Convolutional Neural Network (CNN) model to classify potato plant diseases from leaf images.

🚀 Overview
This project uses a simple CNN architecture to detect and classify diseases in potato leaves. It helps identify common issues like Early Blight, Late Blight, and Healthy leaves, based on image data.

📁 Dataset
The dataset used is from the PlantVillage dataset, specifically filtered for potato leaf images.

Classes:

Potato__Early_blight

Potato__Late_blight

Potato__Healthy

🧠 Model
The CNN consists of multiple convolutional and pooling layers, followed by dense layers for classification.

Key Features:

Input Shape: 128x128 RGB images

Activation: ReLU & Softmax

Loss: Categorical Crossentropy

Optimizer: Adam

🛠️ Tech Stack
Python

TensorFlow / Keras

NumPy, Matplotlib

Jupyter Notebook

🧪 How to Run
Clone the repo

bash
Copy
Edit
git clone https://github.com/your-username/potato-disease-classifier.git
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the Jupyter notebook

bash
Copy
Edit
jupyter notebook Potato_Disease_Classifier.ipynb
📈 Results
Achieved over 90% accuracy on validation data after training for a few epochs.

📝 Future Work
Improve accuracy using transfer learning (e.g., MobileNet, ResNet)

Build a web/mobile app interface for real-time detection

Expand to other crops and diseases

🙋‍♂️ Author
Krish Patel – LinkedIn | Twitter | Portfolio
