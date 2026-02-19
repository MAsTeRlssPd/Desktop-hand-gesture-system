import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

CSV_FILE = "gesture_dataset.csv"
MODEL_FILE = "gesture_model.pkl"

def main():
    print("Loading dataset...")
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"Error: {CSV_FILE} not found. Please run collect_data.py first.")
        return

    # Split features (X) and labels (y)
    X = df.drop('label', axis=1)
    y = df['label']

    print(f"Found {len(df)} samples across classes: {y.unique()}")

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    print("Training model...")
    model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=None, min_samples_leaf=2)
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy on Test Data: {accuracy * 100:.2f}%")

    # Save the model
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved successfully to {MODEL_FILE}")

if __name__ == "__main__":
    main()