import pandas as pd
import requests
import json
from tqdm import tqdm
import numpy as np

# Load your test data
df_test = pd.read_csv("../data/hf_models_withmodelcard_nov2024.csv")
X_test = df_test[["task_group", "author_category", "language_category", "location"]]
y_true = df_test["downloads_category"].astype(str).str.strip().tolist()


print("="*50)
print(f"Downloads category: {df_test['downloads_category'].value_counts()}")
print(f"X_test shape: {X_test.shape}")
print(f"y_true shape: {len(y_true)}") 
print(f"Sample Y_true: {y_true[:10]}")
print("="*50)

# Settings
BATCH_SIZE = 10000
URL = "http://localhost:8080/predict"
HEADERS = {"Content-Type": "application/json"}

# Store all predictions
all_preds = []

# Batch and send requests
for i in tqdm(range(0, len(X_test), BATCH_SIZE)):

    print("="*50)
    print(f"Processing batch {i // BATCH_SIZE + 1}...")
    print(f"Batch size: {min(BATCH_SIZE, len(X_test) - i)}")
    print(f"Batch indices: {i} to {min(i + BATCH_SIZE, len(X_test))}")
    print("="*50)


    batch_data = X_test.iloc[i:i+BATCH_SIZE].to_dict(orient="records")
    request_payload = {"instances": batch_data}
    
    response = requests.post(URL, headers=HEADERS, data=json.dumps(request_payload))
    response.raise_for_status()
    
    predictions = response.json()["predictions"]
    
    # Extract the class with the highest score for each prediction
    batch_preds = [pred["classes"][np.argmax(pred["scores"])] for pred in predictions]
    all_preds.extend(batch_preds)

    print("="*50)
    print(f"Batch {i // BATCH_SIZE + 1} processed.")
    print("="*50)

# Evaluate accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_true[:len(all_preds)], all_preds)
print(f"Accuracy: {accuracy:.4f}")
