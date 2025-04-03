import time
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Configure logging
logging.basicConfig(filename='training_logs.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

def train_and_evaluate_model(df, features, target, algorithm):
    """
    Updated function with categorical encoding
    """
    X = df[features]
    y = df[target]

    # Encode target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), features)
        ])
    
    # Create model pipeline
    if algorithm == 'logistic':
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000))
        ])
    elif algorithm == 'random_forest':
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier())
        ])
    else:
        raise ValueError("Invalid algorithm name. Choose 'logistic' or 'random_forest'.")

    # Training and evaluation
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    start_time = time.time()
    y_pred = model.predict(X_test)
    test_time = time.time() - start_time

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    micro_f1 = f1_score(y_test, y_pred, average='micro')
    macro_f1 = f1_score(y_test, y_pred, average='macro')


    # Print and log results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Micro Average F1 Score: {micro_f1:.4f}")
    print(f"Macro Average F1 Score: {macro_f1:.4f}")
    print(f"Training Time: {train_time:.2f} seconds")
    print(f"Testing Time: {test_time:.2f} seconds")

    logging.info(f"Algorithm: {algorithm}")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"Micro Average F1 Score: {micro_f1:.4f}")
    logging.info(f"Macro Average F1 Score: {macro_f1:.4f}")
    logging.info(f"Training Time: {train_time:.2f} seconds")
    logging.info(f"Testing Time: {test_time:.2f} seconds")

import matplotlib.pyplot as plt
import seaborn as sns

def create_distrib(df):
    print(df.downloads_category.value_counts())
    # Create a count plot
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")

    # Create grouped bar chart
    ax = sns.countplot(
        x="task_group",
        hue="downloads_category",
        data=df,
        order=df['task_group'].value_counts().index,
        palette="viridis"
    )

    # Customize plot
    plt.title("Distribution of Download Categories Across Task Groups", fontsize=16, pad=20)
    plt.xlabel("Task Group", fontsize=12)
    plt.ylabel("Count of Models", fontsize=12)
    plt.xticks(rotation=45, ha="right")

    # Add percentage annotations
    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 3,
                f'{height/total:.1%}',
                ha="center", fontsize=9)

    plt.legend(title="Download Category", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def percent_distrib(df):
    categorical_columns = ['task_group', 'author_category', 'language_category', 'downloads_category', 'location']

    for col in categorical_columns:
        print(f"Percentage distribution for {col}:")
        percentages = df[col].value_counts(normalize=True) * 100
        print(percentages.round(2))
        print("\n")

def run_advanced_smote(df, features, target):
    """
    This function applies SMOTE to the dataset and returns the resampled DataFrame.
    """
    from imblearn.over_sampling import SMOTE
    from collections import Counter

    # Separate features and target
    X = df[features]
    y = df[target]

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Create a new DataFrame with the resampled data
    resampled_df = pd.DataFrame(X_resampled, columns=features)
    resampled_df[target] = y_resampled

    # Print the original and new class distribution
    print(f"Original class distribution: {Counter(y)}")
    print(f"Resampled class distribution: {Counter(y_resampled)}")

    return resampled_df