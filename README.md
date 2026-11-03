Customer Churn Prediction using ANN

An end-to-end Customer Churn Prediction project that uses an
Artificial Neural Network (ANN) to estimate whether a bank customer is
likely to leave the bank. The project includes data preprocessing, model
training, saved preprocessing objects, and an interactive Streamlit
application for making predictions.

Project Overview

Customer churn occurs when a customer stops using a company's products
or services. Predicting churn can help businesses identify customers who
may be at risk and take steps to improve retention.

This project uses the Churn Modelling dataset and a neural-network
classification model to predict customer churn from customer and account
information.

Features

Data preprocessing and feature encoding

Feature scaling using StandardScaler

Artificial Neural Network built with TensorFlow/Keras

Model and preprocessing objects saved for reuse

Interactive prediction interface built with Streamlit

Jupyter notebooks for experimentation and prediction

Technology Stack

Language: Python

Deep Learning: TensorFlow / Keras

Data processing: Pandas, NumPy

Machine learning utilities: Scikit-learn

Model persistence: Joblib

Web application: Streamlit

Development: Jupyter Notebook / Python

Repository Structure

Customer-Churn-Prediction-ANN/
│
├── app.py                    # Streamlit prediction application
├── train.py                  # Data preprocessing and ANN training
├── model.h5                  # Trained ANN model
├── label_encoder.pkl         # Saved label encoder
├── one_hot_encoder.pkl       # Saved one-hot encoder
├── scaler.pkl                # Saved feature scaler
├── requirements.txt          # Python dependencies
│
├── data/
│   └── Churn_Modelling.csv   # Dataset
│
├── experiments.ipynb         # Model experimentation
└── prediction.ipynb          # Prediction notebook

Model Architecture

The ANN model in this project uses the following dense-layer structure:

Layer                    Units Activation

Input / hidden layer        64 ReLU
Hidden layer                32 ReLU
Output layer                 1 Sigmoid

The model is trained as a binary classification model using the Adam
optimizer and binary cross-entropy loss. Accuracy is used as a
training metric. The training script also uses Early Stopping and
TensorBoard callbacks.

Data Preprocessing

The training workflow includes:

Load the Churn Modelling CSV dataset.

Encode the Gender column using LabelEncoder.

Encode the Geography column using OneHotEncoder.

Scale numerical features using StandardScaler.

Train the ANN model for binary churn classification.

Save the trained model and preprocessing objects for use by the
application.

Important: The preprocessing steps and feature order used by the
Streamlit application must match the steps and feature order used during
training.

Installation

1. Clone the repository

git clone https://github.com/gauravkatre55/Customer-Churn-Prediction-ANN.git
cd Customer-Churn-Prediction-ANN

2. Create and activate a virtual environment

Windows PowerShell:

python -m venv venv
.\venv\Scripts\Activate.ps1

If PowerShell blocks activation, you can use Command Prompt instead:

venv\Scripts\activate.bat

3. Install dependencies

pip install -r requirements.txt

Run the Streamlit Application

From the project root directory, run:

streamlit run app.py

Streamlit will display a local address in the terminal. Open that
address in your browser to use the application.

Train the Model Again

If you want to retrain the model, run:

python train.py

The training script should generate or update the model and
preprocessing files used by the application. Keep the saved
preprocessing objects consistent with the model.

Saved Model and Preprocessing Files

The repository includes these saved artifacts:

model.h5 --- trained ANN model

label_encoder.pkl --- label encoder

one_hot_encoder.pkl --- one-hot encoder

scaler.pkl --- feature scaler

These files allow the application to load the trained model and apply
the saved preprocessing steps when a prediction is requested.

Notes

Install the dependencies listed in requirements.txt before running
the application.

Use compatible TensorFlow/Keras and scikit-learn versions when
loading saved model and preprocessing files.

The venv/ environment and TensorBoard logs/ directory are
excluded from Git because they are local/generated files.

This project is an educational machine-learning application.
Predictions should be evaluated and validated before being used for
real business decisions.

Author

Gaurav Katre

GitHub: @gauravkatre55
