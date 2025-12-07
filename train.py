import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TimeDistributed, GlobalAveragePooling2D, LSTM, Dense
from dataset_tf import get_datasets

# --------------------------
# 1) Load your dataset
# --------------------------
X_train, X_test, y_train, y_test = get_datasets()

# One-hot encode labels (0–3)
y_train = tf.keras.utils.to_categorical(y_train, 4)
y_test  = tf.keras.utils.to_categorical(y_test, 4)

print("\nDataset ready!")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# --------------------------
# 2) Build CNN–LSTM model
# --------------------------
def build_model():
    base_cnn = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(128, 128, 3)
    )
    base_cnn.trainable = False  # keep it very simple

    inp = Input(shape=(15, 128, 128, 3))

    x = TimeDistributed(base_cnn)(inp)
    x = TimeDistributed(GlobalAveragePooling2D())(x)

    x = LSTM(64)(x)

    x = Dense(32, activation="relu")(x)
    out = Dense(4, activation="softmax")(x)

    model = Model(inp, out)
    return model


# --------------------------
# 3) Train with 3 optimizers
# --------------------------
optimizers = {
    "SGD": tf.keras.optimizers.SGD(learning_rate=0.001),
    "Adam": tf.keras.optimizers.Adam(learning_rate=0.0005),
    "Adagrad": tf.keras.optimizers.Adagrad(learning_rate=0.001)
}

for name, opt in optimizers.items():
    print(f"\n==============================")
    print(f"Training with {name}")
    print(f"==============================\n")

    model = build_model()
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

    history = model.fit(
        X_train, y_train,
        epochs=10,              # keep it simple
        batch_size=8,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # Print final accuracy
    final_train_acc = history.history["accuracy"][-1]
    final_val_acc   = history.history["val_accuracy"][-1]

    print(f"\n{name} results:")
    print(f"  Final Training Accuracy: {final_train_acc:.4f}")
    print(f"  Final Validation Accuracy: {final_val_acc:.4f}")

    # Save model
    model.save(f"model_{name}.h5")
    print(f"Saved model as model_{name}.h5\n")
