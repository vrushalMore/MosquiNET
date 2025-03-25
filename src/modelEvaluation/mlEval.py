import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def load_data():
    df = pd.read_csv("../../data/raw/data.csv")
    X = df.drop(columns=['Habitable', 'Area Name'])
    y = df['Habitable']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def preprocess(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)

def train_and_evaluate(models, X_train, y_train, X_test, y_test):
    best_model = None
    best_f1 = 0.0
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"{name}:")
        print("  Accuracy:", accuracy_score(y_test, y_pred))
        print("  Precision:", precision_score(y_test, y_pred, average='weighted'))
        print("  Recall:", recall_score(y_test, y_pred, average='weighted'))
        print("  F1 Score:", f1)
        print("-" * 40)

        if f1 > best_f1:
            best_f1 = f1
            best_model = model

    if best_model:
        joblib.dump(best_model, "mlModel.pkl")
    
X_train, X_test, y_train, y_test = load_data()
X_train, X_test = preprocess(X_train, X_test)

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier()
}

train_and_evaluate(models, X_train, y_train, X_test, y_test)
