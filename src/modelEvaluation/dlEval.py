import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_data():
    df = pd.read_csv("../../data/raw/data.csv")
    X = df.drop(columns=['Habitable', 'Area Name'])
    y = df['Habitable']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def preprocess(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)

def build_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)
    y_pred = (model.predict(X_test) > 0.5).astype(int)

    f1 = f1_score(y_test, y_pred, average='weighted')

    print("Deep Learning Model:")
    print("  Accuracy:", accuracy_score(y_test, y_pred))
    print("  Precision:", precision_score(y_test, y_pred, average='weighted'))
    print("  Recall:", recall_score(y_test, y_pred, average='weighted'))
    print("  F1 Score:", f1)

    return f1, model

X_train, X_test, y_train, y_test = load_data()
X_train, X_test = preprocess(X_train, X_test)

best_f1 = 0.0
best_model = None

for i in range(3):  
    model = build_model(X_train.shape[1])
    f1, trained_model = train_and_evaluate(model, X_train, y_train, X_test, y_test)

    if f1 > best_f1:
        best_f1 = f1
        best_model = trained_model

if best_model:
    with open("dlModel.pkl", "wb") as f:
        pickle.dump(best_model, f)