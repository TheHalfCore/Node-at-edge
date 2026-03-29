import pandas as pd
import os, sys
sys.path.insert(0, '..') 
import lib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt

train = pd.read_csv("./data/UniMiB/unimib_train.csv")
val   = pd.read_csv("./data/UniMiB/unimib_val.csv")
test  = pd.read_csv("./data/UniMiB/unimib_test.csv")

print(train.shape, val.shape, test.shape) # (number of rows, number of columns) for each dataset
print(train.columns) # List of column names in the training dataset

def aggregate_windows(df): # Aggregate sensor data by window (ID)
    agg = df.groupby("ID").agg({ # Aggregate statistics for each sensor axis and magnitude
        "ax": ["mean", "std", "min", "max"],
        "ay": ["mean", "std", "min", "max"],
        "az": ["mean", "std", "min", "max"],
        "mag": ["mean", "std", "min", "max"],
        "label": "first"
    })

    # Flatten column names
    agg.columns = [
        "_".join(col) if isinstance(col, tuple) else col
        for col in agg.columns
    ]

    return agg.reset_index(drop=True)


train_agg = aggregate_windows(train)
val_agg   = aggregate_windows(val)
test_agg  = aggregate_windows(test)

print(train_agg.shape)

X_train = train_agg.drop(columns=["label_first"]) # Drop the label column for training
y_train = train_agg["label_first"] # Extract the label column for training

X_val = val_agg.drop(columns=["label_first"]) # Drop the label column for validation
y_val = val_agg["label_first"] # Extract the label column for validation

X_test = test_agg.drop(columns=["label_first"]) # Drop the label column for testing
y_test = test_agg["label_first"] # Extract the label column for testing

rf = RandomForestClassifier(
    n_estimators=8,     # match NODE: 8 trees
    max_depth=2,        # match NODE depth
    max_features=None,  # use all features (like NODE)
    bootstrap=True,     # use bootstrap samples (like NODE)
    random_state=42,    # for reproducibility
    n_jobs=-1           # use all CPU cores for training
)

rf.fit(X_train, y_train)

# Validation
y_val_pred = rf.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))

# Test
y_test_pred = rf.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))

print("\nClassification Report (Test):")
print(classification_report(y_test, y_test_pred))

print("F1 (weighted):", f1_score(y_test, y_test_pred, average="weighted"))
print("F1 (macro):   ", f1_score(y_test, y_test_pred, average="macro"))
print("F1 (micro):   ", f1_score(y_test, y_test_pred, average="micro"))

cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(6,6))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.colorbar()
plt.show()