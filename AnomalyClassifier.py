import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from packaging.version import parse as parse_version
from joblib import dump  

def load_prepared_data():
    """
    Load the prepared full and review datasets from CSV files.
    These files are generated from data_preparation.py.
    
    Returns:
        full_data (DataFrame): Unlabeled patient dataset.
        review_data (DataFrame): Labeled (review/anomalous) patient dataset.
    """
    full_data = pd.read_csv('full_dataset_prepared.csv')
    review_data = pd.read_csv('review_dataset_prepared.csv')
    return full_data, review_data

def create_training_sample(full_data, review_data, sample_frac=0.1):
    """
    Create a training sample by randomly sampling a fraction of the full (unlabeled) dataset
    and concatenating it with the review (anomalous) dataset.
    
    Parameters:
        full_data (DataFrame): The unlabeled dataset.
        review_data (DataFrame): The dataset with labeled anomalies.
        sample_frac (float): Fraction of full_data to sample.
    
    Returns:
        train_df (DataFrame): The combined training dataset.
    """
    sample_df = full_data.sample(frac=sample_frac, random_state=42)
    train_df = pd.concat([sample_df, review_data], axis=0)  # Row-wise concatenation
    return train_df

def oversample_train_data(train_df, feature_cols, target_col='REVIEW_IND'):
    """
    Split the training sample into training and test sets, then use SMOTE to oversample
    the minority class (anomalies) in the training set.
    
    Parameters:
        train_df (DataFrame): Combined training dataset.
        feature_cols (list): List of feature column names.
        target_col (str): Name of the target column.
    
    Returns:
        X_train_res, X_test, y_train_res, y_test: Oversampled training data, test data, 
                                                   training labels, and test labels.
    """
    X = train_df[feature_cols]
    y = train_df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    sm = SMOTE(random_state=123, sampling_strategy=0.5)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    return X_train_res, X_test, y_train_res, y_test

def train_random_forest(X_train, y_train, n_estimators=50, random_state=123):
    """
    Train a Random Forest classifier on the oversampled training data.
    
    Parameters:
        X_train (DataFrame): Training features.
        y_train (Series): Training target.
        n_estimators (int): Number of trees in the forest.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        clf (RandomForestClassifier): The trained classifier.
    """
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train, y_train)
    return clf

def predict_anomalies(clf, df, feature_cols, threshold):
    """
    Predict anomaly probabilities for the full dataset using the trained classifier.
    Flag patients as anomalous if their predicted probability exceeds the given threshold.
    
    Parameters:
        clf (classifier): The trained Random Forest classifier.
        df (DataFrame): Full dataset for prediction.
        feature_cols (list): List of feature column names.
        threshold (float): Probability threshold for flagging anomalies.
    
    Returns:
        df (DataFrame): The input DataFrame with two additional columns:
                        'prob_true' - the predicted probability for anomalies,
                        'predicted' - binary indicator (1 if anomalous, 0 otherwise).
    """
    probs = clf.predict_proba(df[feature_cols])
    df = df.copy()  # Avoid modifying the original DataFrame
    df['prob_true'] = probs[:, 1]
    df['predicted'] = (df['prob_true'] > threshold).astype(int)
    return df

def main():
    # --- Step 1: Load Prepared Data ---
    full_data, review_data = load_prepared_data()
    
    # --- Step 2: Define Feature Columns ---
    # Exclude 'id' and 'REVIEW_IND' from features
    exclude_cols = ['id', 'REVIEW_IND']
    feature_cols = [col for col in full_data.columns if col not in exclude_cols]
    
    # --- Step 3: Create a Training Sample ---
    # Randomly sample a fraction of the full data and combine with review data
    train_df = create_training_sample(full_data, review_data, sample_frac=0.1)
    
    # --- Step 4: Oversample Training Data with SMOTE ---
    X_train, X_test, y_train, y_test = oversample_train_data(train_df, feature_cols, target_col='REVIEW_IND')
    
    # --- Step 5: Train the Random Forest Classifier ---
    clf = train_random_forest(X_train, y_train, n_estimators=50)
    
    # --- Step 6: Evaluate the Model on the Test Set ---
    y_pred = clf.predict(X_test)
    train_accuracy = clf.score(X_train, y_train)
    test_accuracy = clf.score(X_test, y_test)
    print("Training Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)
    
    # Generate classification report and print it
    report = classification_report(y_test, y_pred, output_dict=True)
    print("\nClassification Report:")
    print(report)
    
    # Save the classification report to a CSV file
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv("classification_report.csv", index=True)
    print("Classification report saved as classification_report.csv")
    
    # --- Step 7: Save the Trained Model ---
    dump(clf, 'anomalyclassifier.pkl')
    print("Model saved as anomalyclassifier.pkl")
    
    # --- Step 8: Predict Anomalies on the Full Dataset ---
    anomalous_df = predict_anomalies(clf, full_data, feature_cols, threshold=0.5)
    num_anomalies = anomalous_df[anomalous_df['predicted'] == 1].shape[0]
    print("Number of anomalous patients detected:", num_anomalies)
    
    # --- Step 9: Plot and Save the Distribution of Predicted Anomaly Probabilities ---
    probs_full = clf.predict_proba(full_data[feature_cols])[:, 1]
    plt.figure(figsize=(8,6))
    plt.hist(probs_full, bins=50, color='skyblue', edgecolor='black')
    plt.xlabel("Predicted Probability for Anomaly")
    plt.ylabel("Frequency")
    plt.title("Distribution of Anomaly Probabilities in Full Dataset")
    plt.savefig("anomaly_probability_distribution.png")# Save the plot as an image file
    plt.show()
    print("Distribution plot saved as anomaly_probability_distribution.png")

if __name__ == '__main__':
    main()
