# ------------------simple implementation------------------
import pandas as pd
import sys
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
'''
# Check the execution environment to determine the test file source
if 'ipykernel' in sys.modules:
    test_data_path = 'Customer_test.csv'  # Used in Jupyter
else:
    import argparse
    parser = argparse.ArgumentParser(description="One-vs-All SVM Classifier for Customer Segmentation")
    parser.add_argument("test_file", help="Specify the path to the test CSV file.")
    args = parser.parse_args()
    test_data_path = args.test_file

# Load datasets
train_data = pd.read_csv('Customer_train.csv')
test_data = pd.read_csv(test_data_path)

# Remove rows with missing values in the training dataset
train_data.dropna(inplace=True)

# Define features and target variable
features = train_data.drop(columns='Segmentation')
target = train_data['Segmentation']

# Distinguish between numeric and categorical columns
numerical_features = features.select_dtypes(include=['int64', 'float64']).columns
categorical_features = features.select_dtypes(include=['object']).columns

# Standardize numeric features
scaler = StandardScaler()
features[numerical_features] = scaler.fit_transform(features[numerical_features])

# Encode binary categorical features using Label Encoding
encoder = LabelEncoder()
for col in ['Gender', 'Ever_Married']:  # Specify binary categorical columns
    features[col] = encoder.fit_transform(features[col])

# Convert remaining categorical variables into dummy/indicator variables
features = pd.get_dummies(features, columns=[col for col in categorical_features if col not in ['Gender', 'Ever_Married']])

# Split the training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=31)

# Initialize One-vs-All (OvA) SVM classifiers
classifiers = {}
unique_labels = target.unique()

for label in unique_labels:
    binary_labels = (y_train == label).astype(int)  # Create binary target variable
    classifier = svm.SVC(kernel='linear', probability=True)
    classifier.fit(X_train, binary_labels)  # Train the model
    classifiers[label] = classifier

# Make predictions on the validation set
predictions = []

for index in range(len(X_val)):
    # Get the probability for each class
    probabilities = {cls: svc.predict_proba(X_val.iloc[index:index + 1])[0][1] for cls, svc in classifiers.items()}
    predicted_label = max(probabilities, key=probabilities.get)  # Class with highest probability
    predictions.append(predicted_label)

# Calculate accuracy and print the results
accuracy = accuracy_score(y_val, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the predictions to a CSV file
output_df = pd.DataFrame({'predicted': predictions})
output_df.to_csv('ova_predictions.csv', index=False)

# Generate and display confusion matrix and classification report
conf_matrix = confusion_matrix(y_val, predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Plot the confusion matrix using seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
plt.title('Confusion Matrix - One-vs-One SVM')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

classification_rep = classification_report(y_val, predictions, target_names=unique_labels)
print("Classification Report:")
print(classification_rep)

# Plotting the classification report
report = classification_report(y_val, predictions, target_names=unique_labels, output_dict=True)

# Convert the classification report into a DataFrame for visualization
report_df = pd.DataFrame(report).transpose()

# Plotting Precision, Recall, and F1-Score side by side
plt.figure(figsize=(12, 6))

# Preparing data for plotting
metrics = ['precision', 'recall', 'f1-score']
report_df = report_df[:-3]  # Exclude 'accuracy', 'macro avg', and 'weighted avg'

report_df['class'] = report_df.index

# Melting the DataFrame to long format for easier plotting
report_long = pd.melt(report_df, id_vars='class', value_vars=metrics, var_name='metric', value_name='score')

# Plot
sns.barplot(x='class', y='score', hue='metric', data=report_long, palette='Set1')

plt.title('Classification Report: Precision, Recall, F1-Score')
plt.xticks(rotation=45)
plt.xlabel('Class')
plt.ylabel('Score')
plt.legend(loc='upper right', title='Metric')
plt.tight_layout()
plt.show()
'''

# Using SMOTE, we have increased the accuracy to 43%
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Check the execution environment to determine the test file source
if 'ipykernel' in sys.modules:
    test_data_path = 'Customer_test.csv'  # Used in Jupyter
else:
    import argparse
    parser = argparse.ArgumentParser(description="One-vs-All SVM Classifier for Customer Segmentation")
    parser.add_argument("test_file", help="Specify the path to the test CSV file.")
    args = parser.parse_args()
    test_data_path = args.test_file

# Load datasets
train_data = pd.read_csv('Customer_train.csv')
test_data = pd.read_csv(test_data_path)

# Drop rows with NaN values
train_data.dropna(inplace=True)

# Prepare the features and target
X = train_data.drop('Segmentation', axis=1)
y = train_data['Segmentation']

# Separate numerical and categorical columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Scale the numerical features
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Label encode the categorical variables
label_encoder = LabelEncoder()
for col in ['Gender', 'Ever_Married']:
    X[col] = label_encoder.fit_transform(X[col])

# One-hot encode remaining categorical columns
X = pd.get_dummies(X, columns=[col for col in categorical_cols if col not in ['Gender', 'Ever_Married']])

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=31)
X_res, y_res = smote.fit_resample(X, y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=31)

# Train SVM with class_weight='balanced' to handle imbalance
svc = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, class_weight='balanced')
svc.fit(X_train, y_train)

# Predict for the test data
yhat_test = svc.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, yhat_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save predictions to a CSV file
predictions_df = pd.DataFrame(data={'predicted': yhat_test})
predictions_df.to_csv('ova_42.csv', index=False)

# Print confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, yhat_test)
print("Confusion Matrix:")
print(conf_matrix)

report = classification_report(y_test, yhat_test)
print("Classification Report:")
print(report)

# Train SVM with class_weight='balanced' to handle imbalance (with all of training data)
svc = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, class_weight='balanced')
svc.fit(X_res, y_res)

# Predict for the test data
yhat_test = svc.predict(X_test)

# Save the predictions to a CSV file
output_df = pd.DataFrame({'predicted': yhat_test})
output_df.to_csv('ova.csv', index=False)

# unique_labels = y_res.unique()
# # Plot the confusion matrix using seaborn
# plt.figure(figsize=(10, 7))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
# plt.title('Confusion Matrix - One-vs-All SVM')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()

# # Plotting the classification report
# report = classification_report(y_test, yhat_test, target_names=unique_labels, output_dict=True)

# # Convert the classification report into a DataFrame for visualization
# report_df = pd.DataFrame(report).transpose()

# # Plotting Precision, Recall, and F1-Score side by side
# plt.figure(figsize=(12, 6))

# # Preparing data for plotting
# metrics = ['precision', 'recall', 'f1-score']
# report_df = report_df[:-3]  # Exclude 'accuracy', 'macro avg', and 'weighted avg'

# report_df['class'] = report_df.index

# # Melting the DataFrame to long format for easier plotting
# report_long = pd.melt(report_df, id_vars='class', value_vars=metrics, var_name='metric', value_name='score')

# # Plot
# sns.barplot(x='class', y='score', hue='metric', data=report_long, palette='Set1')

# plt.title('Classification Report: Precision, Recall, F1-Score')
# plt.xticks(rotation=45)
# plt.xlabel('Class')
# plt.ylabel('Score')
# plt.legend(loc='upper right', title='Metric')
# plt.tight_layout()
# plt.show()