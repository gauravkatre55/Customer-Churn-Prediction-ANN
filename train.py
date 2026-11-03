import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder ,OneHotEncoder

from sklearn.metrics import accuracy_score, classification_report
# load the dataset
df=pd.read_csv('data/Churn_Modelling.csv')  # Load your dataset

print(df.head())  # Display the first few rows of the dataset

#Label encoding for 'Gender' column
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])

#One hot encoding for 'Geography' column
#One hot encoding for 'Geography' column
one_hot_encoded =OneHotEncoder()
encoded_data = one_hot_encoded.fit_transform(df[['Geography']]).toarray()  # Convert sparse matrix to array for easier handling

one_hot_encoded.get_feature_names_out(['Geography'])
encoded_df = pd.DataFrame(encoded_data, columns=one_hot_encoded.get_feature_names_out(['Geography']))

## concatenate the encoded DataFrame with the original DataFrame (excluding the 'Geography' column)
df=pd.concat([df.drop('Geography', axis=1), encoded_df], axis=1)

## Save the enocoders and scaler

with open('label_encoder.pkl', 'wb') as f:
    joblib.dump(label_encoder, f)
    
with open('one_hot_encoder.pkl', 'wb') as f:
    joblib.dump(one_hot_encoded, f) 
    
## define features and target variable
x=df.drop('Exited', axis=1)  # Features
y=df['Exited']  # Target variable

## train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

## Standardize the features
scaler = StandardScaler()
x_train_sc = scaler.fit_transform(x_train)
x_test_sc = scaler.transform(x_test)
x_train_sc

# Save the scaler for future use
with open('scaler.pkl', 'wb') as f:
    joblib.dump(scaler, f)
    
## ANN Implementation


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
import datetime

(x_train_sc.shape[1])

# Build the ANN model
model=Sequential([
    Dense(64, activation='relu', input_shape=(x_train_sc.shape[1],)), # Hidden Layer 1 connected with input layer
    Dense(32, activation='relu'),# Hidden Layer 2
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

model.summary()

import tensorflow 
opt=tensorflow.keras.optimizers.Adam(learning_rate=0.001)  # Define the optimizer with a learning rate of 0.001
loss=tensorflow.keras.losses.BinaryCrossentropy()  # Define the loss function for binary classification

## Compile the model
model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

## Setup the TensorBoard 
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping
tensorboard_callback = TensorBoard(log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1)

## setup early stopping 
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

### Train the model
histrory=model.fit(x_train_sc, y_train, epochs=100,validation_data=(x_test_sc,y_test), callbacks=[tensorboard_callback, early_stopping])

model.save('model.h5')  # Save the trained model to a file

## Load TensorBoard Extension
%load_ext tensorboard
%tensorboard --logdir logs/fit