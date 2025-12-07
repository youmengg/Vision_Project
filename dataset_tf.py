import os
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

# Folder where Step 2 stored all sequences
BASE = Path(r"C:\Users\marya\Downloads\vision_project\sequences")

def load_all_sequences():
    X = []
    Y = []

    files = os.listdir(BASE)

    for f in files:
        if not f.endswith(".npz"):
            continue

        path = os.path.join(BASE, f)

        data = np.load(path)
        X.append(data["x"])   # shape: (15,128,128,3)
        Y.append(data["y"])   # 0,1,2,3

    X = np.array(X)
    Y = np.array(Y)

    print("Loaded sequences:", X.shape)
    print("Loaded labels:", Y.shape)

    return X, Y


def get_datasets(test_size=0.2):
    X, Y = load_all_sequences()

    # 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_size, random_state=42, shuffle=True
    )

    print("\nTrain shapes:")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)

    print("\nTest shapes:")
    print("X_test:", X_test.shape, "y_test:", y_test.shape)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    get_datasets()
