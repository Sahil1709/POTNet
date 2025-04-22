import pandas as pd
import numpy as np
import smote_variants as sv
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('../data/hf_models_withmodelcard_nov2024.csv')[['task_group', 'author_category', 'language_category', 'location', 'downloads_category']]

# Define target and features
target = 'downloads_category'
features = ['task_group', 'author_category', 'language_category', 'location']

def preprocess_data(df, features, target):
    """
    Preprocess the data:
     - One-hot encode the features to get numeric columns.
     - Label encode the target if necessary.
     - Return processed X and y along with the LabelEncoder.
    """
    # One-hot encode the predictor columns
    X = pd.get_dummies(df[features], drop_first=True)
    
    # Label encode the target column if it's not numeric
    le = LabelEncoder()
    y = le.fit_transform(df[target])

    return X, y, le

def invert_one_hot(X, original_df, features):
    """
    Convert one-hot encoded features back to their original categorical values.
    """
    inverted_df = pd.DataFrame(index=X.index)
    for feature in features:
        # Find the one-hot encoded columns for this feature
        one_hot_columns = [col for col in X.columns if col.startswith(f"{feature}_")]
        if one_hot_columns:
            # Extract the original category from the one-hot encoded columns
            inverted_df[feature] = X[one_hot_columns].idxmax(axis=1).str.split('_', 1).str[1]
        else:
            # If the feature was not one-hot encoded, copy it directly
            inverted_df[feature] = original_df[feature]
    return inverted_df

# Function to apply KMeans SMOTE on a pair of classes
def apply_kmeans_smote(df, features, target, minority_label, majority_label):
    """
    Apply KMeans SMOTE on a pair of classes and return the oversampled DataFrame.
    """
    # Filter the DataFrame for the two classes
    df_pair = df[df[target].isin([minority_label, majority_label])].copy()
    
    # Preprocess the data
    X, y, le = preprocess_data(df_pair, features, target)

    X = X.astype(float)

    # Apply KMeans SMOTE
    oversampler = sv.kmeans_SMOTE(random_state=42)
    X_resampled, y_resampled = oversampler.sample(X.values, y)
    
    # Convert the resampled data back to a DataFrame
    X_resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    y_resampled_df = pd.DataFrame(le.inverse_transform(y_resampled), columns=[target])
    
    print(f"Resampled class distribution for Minority: {minority_label} Majority: {majority_label}:")
    print(np.unique(y_resampled_df, return_counts=True))

    # Invert one-hot encoding to get the original categorical values
    X_original = invert_one_hot(X_resampled_df, df_pair, features)
    
    # Combine the features and target into a single DataFrame
    oversampled_df = pd.concat([X_original, y_resampled_df], axis=1)
    return oversampled_df

# Apply KMeans SMOTE for each pair of classes
oversampled_dfs = []
for label in ['Low', 'Mid', 'High']:
    print(f"Applying KMeans SMOTE for Very Low and {label}...")
    oversampled_df = apply_kmeans_smote(df, features, target, minority_label=label, majority_label='Very Low')
    oversampled_df.to_csv(f"oversampled_{label}.csv", index=False)
    print(f"Saved oversampled DataFrame for Very Low and {label} to oversampled_{label}.csv")
    oversampled_dfs.append(oversampled_df)

# Combine all oversampled subsets into a single DataFrame
final_df = pd.concat(oversampled_dfs, ignore_index=True)

# Print the final class distribution
print(final_df[target].value_counts())

# Save the final DataFrame to a CSV file
final_df.to_csv("kmeans_smote_balanced.csv", index=False)
print("Saved the final oversampled DataFrame to kmeans_smote_balanced.csv")