import pandas as pd
import sys
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from imblearn.over_sampling import SMOTE  # For oversampling
from collections import Counter
import warnings
warnings.filterwarnings("ignore")
'''
# Check the execution environment to determine the test file source
if 'ipykernel' in sys.modules:
    test_data_path = 'Customer_test.csv'  # Used in Jupyter
else:
    import argparse
    parser = argparse.ArgumentParser(description="One-vs-One SVM Classifier for Customer Segmentation")
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
X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.20, random_state=37)

# Initialize One-vs-One (OvO) SVM classifiers for every pair of classes
classifiers = {}
unique_labels = target.unique()

# Build binary classifiers for each pair of classes
for class_pair in itertools.combinations(unique_labels, 2):
    class1, class2 = class_pair
    
    # Filter data for the two classes
    binary_y_train = y_train[(y_train == class1) | (y_train == class2)]
    binary_X_train = X_train[(y_train == class1) | (y_train == class2)]
    
    # Convert labels to binary
    binary_y_train = binary_y_train.map({class1: 0, class2: 1})
    
    # Train the binary classifier
    classifier = svm.SVC(kernel='linear', probability=True)
    classifier.fit(binary_X_train, binary_y_train)
    
    classifiers[class_pair] = classifier

# Predict class by aggregating votes from binary classifiers
predictions = []

for index in range(len(X_val)):
    # Initialize votes for each class
    votes = {label: 0 for label in unique_labels}
    
    for (class1, class2), classifier in classifiers.items():
        prob_class1 = classifier.predict_proba(X_val.iloc[index:index + 1])[0][0]
        prob_class2 = classifier.predict_proba(X_val.iloc[index:index + 1])[0][1]
        
        # Increment vote for the class with higher probability
        if prob_class1 > prob_class2:
            votes[class1] += 1
        else:
            votes[class2] += 1
    
    # Predicted class is the one with the most votes
    predicted_label = max(votes, key=votes.get)
    predictions.append(predicted_label)

# Calculate accuracy and print the results
accuracy = accuracy_score(y_val, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the predictions to a CSV file
output_df = pd.DataFrame({'predicted': predictions})
output_df.to_csv('ovo_predictions.csv', index=False)
print("Predictions saved to 'ovo_predictions.csv'.")

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

# Using SMOTE

# Check the execution environment to determine the test file source
if 'ipykernel' in sys.modules:
    test_data_path = 'Customer_test.csv'  # Used in Jupyter
else:
    import argparse
    parser = argparse.ArgumentParser(description="One-vs-One SVM Classifier for Customer Segmentation with Class Imbalance Handling")
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

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=31)
X_res, y_res = smote.fit_resample(features, target)

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X_res, y_res, test_size=0.2, random_state=31)

# # Split the training data into train and validation sets
# X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=31)

# # Optional: Oversample the minority class using SMOTE
# # Uncomment the following lines if oversampling is required
# smote = SMOTE(random_state=42)
# X_train, y_train = smote.fit_resample(X_train, y_train)
print(f"Resampled class distribution: {Counter(y_train)}")

# Initialize One-vs-One (OvO) SVM classifiers with class weights
classifiers = {}
unique_labels = target.unique()

# Iterate over all pairs of classes for OvO
for class1, class2 in itertools.combinations(unique_labels, 2):
    binary_X_train = X_train[(y_train == class1) | (y_train == class2)]
    binary_y_train = y_train[(y_train == class1) | (y_train == class2)]
    
    # Map classes to 0 and 1 for binary classification
    binary_y_train = binary_y_train.map({class1: 0, class2: 1})

    # Train SVM with class weights
    classifier = svm.SVC(kernel='linear', probability=True, class_weight='balanced')
    classifier.fit(binary_X_train, binary_y_train)

    classifiers[(class1, class2)] = classifier

# Make predictions on the validation set
predictions = []
for index in range(len(X_val)):
    class_votes = {label: 0 for label in unique_labels}

    # Iterate over classifiers and accumulate votes for each class
    for (class1, class2), binary_classifier in classifiers.items():
        prob_class1, prob_class2 = binary_classifier.predict_proba(X_val.iloc[index:index + 1])[0]
        if prob_class1 > prob_class2:
            class_votes[class1] += 1
        else:
            class_votes[class2] += 1

    # Select the class with the most votes
    predicted_label = max(class_votes, key=class_votes.get)
    predictions.append(predicted_label)

# Calculate accuracy and print the results
accuracy = accuracy_score(y_val, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the predictions to a CSV file
# output_df = pd.DataFrame({'predicted': predictions})
# output_df.to_csv('ovo_predictions.csv', index=False)

# Generate and display confusion matrix and classification report
conf_matrix = confusion_matrix(y_val, predictions)
print("Confusion Matrix:")
print(conf_matrix)

classification_rep = classification_report(y_val, predictions, target_names=unique_labels)
print("Classification Report:")
print(classification_rep)

# For all of train data
# Iterate over all pairs of classes for OvO
for class1, class2 in itertools.combinations(unique_labels, 2):
    binary_X_res = X_res[(y_res == class1) | (y_res == class2)]
    binary_y_res = y_res[(y_res == class1) | (y_res == class2)]
    
    # Map classes to 0 and 1 for binary classification
    binary_y_res = binary_y_res.map({class1: 0, class2: 1})

    # res SVM with class weights
    classifier = svm.SVC(kernel='linear', probability=True, class_weight='balanced')
    classifier.fit(binary_X_res, binary_y_res)

    classifiers[(class1, class2)] = classifier

# Make predictions on the validation set
predictions = []
for index in range(len(X_val)):
    class_votes = {label: 0 for label in unique_labels}

    # Iterate over classifiers and accumulate votes for each class
    for (class1, class2), binary_classifier in classifiers.items():
        prob_class1, prob_class2 = binary_classifier.predict_proba(X_val.iloc[index:index + 1])[0]
        if prob_class1 > prob_class2:
            class_votes[class1] += 1
        else:
            class_votes[class2] += 1

    # Select the class with the most votes
    predicted_label = max(class_votes, key=class_votes.get)
    predictions.append(predicted_label)

# Save the predictions to a CSV file
output_df = pd.DataFrame({'predicted': predictions})
output_df.to_csv('ovo.csv', index=False)

# Plot the confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels)
# plt.title("Confusion Matrix")
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.show()

# # Plotting the classification report
# report = classification_report(y_val, predictions, target_names=unique_labels, output_dict=True)

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
