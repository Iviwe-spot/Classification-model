import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_squared_error

# Load the dataset
df = pd.read_csv('iris-one-hot-encoded.csv', header=None)

# Check the structure of the DataFrame
print(df.head())

# Correct the column names based on the number of columns in the DataFrame
df.columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species', 'OneHot1', 'OneHot2', 'OneHot3']

# Extract relevant features and target
X = df[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']].values
y = df[['OneHot1', 'OneHot2', 'OneHot3']].values

# Normalize the input values using min-max normalization
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Define a function to split the dataset
def split_dataset(X, y, train_size, val_size, test_size):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(1 - train_size), random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(test_size / (val_size + test_size)), random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Define a function to train and evaluate a neural network
def train_evaluate_nn(hidden_layers, X_train, X_val, X_test, y_train, y_val, y_test, epochs=1000):
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, activation='logistic', solver='adam', max_iter=1, warm_start=True, verbose=0)
    
    print(f"Training a neural network with hidden layers: {hidden_layers}.")
    for epoch in range(epochs):
        mlp.fit(X_train, y_train)
        y_train_pred = mlp.predict_proba(X_train)
        y_val_pred = mlp.predict_proba(X_val)
        train_loss = mean_squared_error(y_train, y_train_pred)
        val_loss = mean_squared_error(y_val, y_val_pred)
        print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Evaluate the model on the test set
    y_test_pred = mlp.predict(X_test)
    test_accuracy = accuracy_score(y_test.argmax(axis=1), y_test_pred.argmax(axis=1))
    print(f"Test Set Accuracy: {test_accuracy:.4f}")

# Customize the neural network structure and data split percentages
hidden_layers = (20, 50, 90)  # Example: four hidden layers with 8 neurons each
train_size = 0.7
val_size = 0.15
test_size = 0.15
X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X_normalized, y, train_size, val_size, test_size)

# Train and evaluate the neural network
train_evaluate_nn(hidden_layers, X_train, X_val, X_test, y_train, y_val, y_test, epochs=1000)
