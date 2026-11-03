
a456296c-d989-45dc-a939-9d520b33afc1.png
ileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = 'model/model.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

Traceback:

File "C:\Users\gaura\OneDrive\Desktop\DL Pratical\app.py", line 10, in <module>
    model = tf.keras.models.load_model('model/model.h5')
File "C:\Users\gaura\AppData\Roaming\Python\Python313\site-packages\keras\src\saving\saving_api.py", line 197, in load_model return legacy_h5_format.load_model_from_hdf5( ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^ filepath, ^^^^^^^^^ ...<2 lines>... safe_mode=safe_mode, ^^^^^^^^^^^^^^^^^^^^ ) ^

File "C:\Users\gaura\AppData\Roaming\Python\Python313\site-packages\keras\src\legacy\saving\legacy_h5_format.py", line 120, in load_model_from_hdf5 f = h5py.File(filepath, mode="r")

File "C:\Users\gaura\AppData\Roaming\Python\Python313\site-packages\h5py\_hl\files.py", line 564, in __init__ fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)

File "C:\Users\gaura\AppData\Roaming\Python\Python313\site-packages\h5py\_hl\files.py", line 238, in make_fid fid = h5f.open(name, flags, fapl=fapl)

File "h5py/_objects.pyx", line 56, in h5py._objects.with_phil.wrapper

File "h5py/_objects.pyx", line 57, in h5py._objects.with_phil.wrapper

File "h5py/h5f.pyx", line 102, in h5py.h5f.open check it


4fdeb35a-e68b-4a55-a2b3-0922a295e126.png

ed438809-5026-436e-bc14-612c469cd8cb.png

Pasted code.py
Python
also add background prefessional photo

Excellent! ✅ Everything is correct.

Your Git status confirms:

All 11 required project files are staged.

venv/ is not tracked.

No logs/ files are staged.

Your ANN model and preprocessing files are included.

Now let's push your project.

Step 1: Commit
git commit -m "Initial commit - Customer Churn Prediction ANN"
Step 2: Rename branch to main
git branch -M main
Step 3: Check GitHub remote
git ls-remote --heads origin
If this returns no output, continue.

Step 4: Push to GitHub
git push -u origin main
Important: If Step 3 displays an existing branch, stop and send me its output before pushing.

Your repository:


git push -u origin main
Enumerating objects: 14, done.
Counting objects: 100% (14/14), done.
Delta compression using up to 12 threads
Compressing objects: 100% (11/11), done.
Writing objects: 100% (14/14), 308.51 KiB | 38.56 MiB/s, done.
Total 14 (delta 2), reused 10 (delta 2), pack-reused 0 (from 0)
remote: Resolving deltas: 100% (2/2), done.
To gauravkatre55/Customer-Churn-Prediction-ANN.git
 * [new branch]        main -> main
branch 'main' set up to track 'origin/main'.
(base) PS C:\Users\gaura\OneDrive\Desktop\DL Pratical> 

🎉 Congratulations! Your Customer Churn Prediction ANN project has been successfully pushed to GitHub!

Your output confirms:

[new branch] main -> main
branch 'main' set up to track 'origin/main'
There were no errors. Your project is now uploaded.

Your GitHub repository

Final project checklist
Component	Status
Streamlit application (app.py)	✅ Uploaded
ANN model (model.h5)	✅ Uploaded
Label Encoder	✅ Uploaded
One-Hot Encoder	✅ Uploaded
Standard Scaler	✅ Uploaded
Dataset	✅ Uploaded
Training code	✅ Uploaded
Jupyter notebooks	✅ Uploaded
Requirements	✅ Uploaded
Virtual environment	❌ Excluded (correctly)
My recommendation: 3 improvements
README.md: Add project overview, technologies, installation instructions, screenshots and results.

GitHub screenshots: Show your Streamlit application's interface and prediction output.

requirements.txt: Ensure all dependency versions are compatible, especially TensorFlow and scikit-learn.

Your next step should be creating a professional README.md so recruiters can understand your project easily.

I can prepare a complete, professional GitHub README for this ANN Customer Churn Prediction project.


create readme file

Your professional README file is ready!

📄 

Add it to your GitHub repository
Download the file.

Copy README.md into your project folder:
DL Pratical

Open PowerShell in that folder.

Run:

git add README.md
git commit -m "Add professional project README"
git push
Your GitHub repository will display the README automatically on its homepage.

The README includes project overview, ANN architecture, preprocessing, technologies, installation instructions, execution steps and author details. 


You’ve reached your data analysis limit. Upgrade to ChatGPT Go or try again tomorrow after 10:04 AM.
Upgrade to Go

README.md


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

Layer Units Activation

Input / hidden layer 64 ReLU
Hidden layer 32 ReLU
Output layer 1 Sigmoid

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

If you find this project useful, feel free to explore the repository and
share feedback.

