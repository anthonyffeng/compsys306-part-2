import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the test data
X_test, y_test = joblib.load('src_mark/testing_data.joblib')

# Load the trained model (assuming you saved the model using joblib)
model = joblib.load('src_mark/model-anthony.joblib')

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print classification report for precision, recall, and F1-score
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix to see the performance across classes
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))