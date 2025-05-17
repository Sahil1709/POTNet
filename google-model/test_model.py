import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio   # must come before loading the model
from sklearn.metrics import accuracy_score

# path to the directory containing saved_model.pb
MODEL_DIR = "cloud-ai-platform-c541b3e3-934f-414e-9196-8e2bf7a7fb59/model-6131366119153336320/tf-saved-model/2025-05-15T03:23:14.175361Z/predict/001"

# load the model
loaded = tf.saved_model.load(MODEL_DIR)
infer = loaded.signatures["serving_default"]

# read your test set
df_test = pd.read_csv("cloud-ai-platform-c541b3e3-934f-414e-9196-8e2bf7a7fb59/hf_models_withmodelcard_nov2024.csv")
features = ["task_group","author_category","language_category","location"]  # whatever your model signature expects

# build inputs dict
inputs = {}
for col in features:
    # if your model expects string tensors:
    inputs[col] = tf.constant(df_test[col].astype(str).values)
    # or float: tf.constant(df_test[col].values, dtype=tf.float32)

# call the model
outputs = infer(**inputs)

# inspect output keys
print("Available outputs:", list(outputs.keys()))
# typically something like 'output_0' or 'predictions'
pred_tensor = outputs["output_0"]    # replace with the right key

# if it’s logits or probabilities, you might need argmax
if pred_tensor.shape[-1] > 1:
    pred_indices = tf.argmax(pred_tensor, axis=-1).numpy()
else:
    # if binary, maybe threshold
    pred_indices = (pred_tensor.numpy() > 0.5).astype(int).flatten()

# decode indices back to category names if you kept a LabelEncoder
# (assuming you saved it when exporting)
#    from sklearn.preprocessing import LabelEncoder
#    le = load_my_label_encoder()
#    preds = le.inverse_transform(pred_indices)

# attach predictions
df_test["pred_idx"] = pred_indices
print(df_test.head())


y_true = df_test["downloads_category"]
y_pred = df_test["predicted_downloads_category"]  # or after inverse_transform

print("Accuracy:", accuracy_score(y_true, y_pred))