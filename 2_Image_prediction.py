import os
import random

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split

from PIL import Image


# ==========================================================
# 1. Configuration
# ==========================================================

CAT_PATH = os.path.join("PetImages", "Cat")
DOG_PATH = os.path.join("PetImages", "Dog")

MODEL_PATH = "cat_dog_model.keras"

IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 30
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
RANDOM_STATE = 42


# ==========================================================
# 2. Reproducibility
# ==========================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ==========================================================
# 3. Load Dataset
# ==========================================================

def load_images():

    images = []
    labels = []

    # ------------------------------------------------------
    # Cat = 0
    # ------------------------------------------------------

    for image_name in os.listdir(CAT_PATH):

        image_path = os.path.join(CAT_PATH, image_name)

        try:
            image = Image.open(image_path)
            image = image.convert("RGB")
            image = image.resize((IMG_SIZE, IMG_SIZE))

            images.append(np.array(image))
            labels.append(0)

        except (OSError, ValueError) as error:
            print(f"Skipping invalid image: {image_path}")
            print(f"Reason: {error}")

    # ------------------------------------------------------
    # Dog = 1
    # ------------------------------------------------------

    for image_name in os.listdir(DOG_PATH):

        image_path = os.path.join(DOG_PATH, image_name)

        try:
            image = Image.open(image_path)
            image = image.convert("RGB")
            image = image.resize((IMG_SIZE, IMG_SIZE))

            images.append(np.array(image))
            labels.append(1)

        except (OSError, ValueError) as error:
            print(f"Skipping invalid image: {image_path}")
            print(f"Reason: {error}")

    return np.array(images), np.array(labels)


# ==========================================================
# 4. Prepare Dataset
# ==========================================================

X, y = load_images()

print("\nDataset Information")
print("-------------------")
print("Images shape :", X.shape)
print("Labels shape :", y.shape)


# ==========================================================
# 5. Normalize Images
# ==========================================================

X = X.astype("float32") / 255.0


# ==========================================================
# 6. Train-Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nData Split")
print("----------")
print("Training images :", X_train.shape)
print("Testing images  :", X_test.shape)


# ==========================================================
# 7. Create CNN Model
# ==========================================================

def create_model():

    model = Sequential()

    model.add(
        Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
    )

    model.add(MaxPool2D((2, 2)))

    model.add(
        Conv2D(
            64,
            (3, 3),
            activation="relu"
        )
    )

    model.add(MaxPool2D((2, 2)))

    model.add(
        Conv2D(
            128,
            (3, 3),
            activation="relu"
        )
    )

    model.add(MaxPool2D((2, 2)))

    model.add(Flatten())

    model.add(
        Dense(
            128,
            activation="relu"
        )
    )

    model.add(
        Dense(
            64,
            activation="relu"
        )
    )

    model.add(
        Dense(
            1,
            activation="sigmoid"
        )
    )

    return model


model = create_model()


# ==========================================================
# 8. Model Summary
# ==========================================================

model.summary()

# ==========================================================
# 9. Compile Model
# ==========================================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ==========================================================
# 10. Early Stopping
# ==========================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


# ==========================================================
# 11. Train Model
# ==========================================================

print("\nStarting Training...")
print("--------------------")

history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SIZE,
    callbacks=[early_stop],
    verbose=1
)


# ==========================================================
# 12. Plot Accuracy
# ==========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("CNN Training vs Validation Accuracy")
plt.legend()

plt.show()


# ==========================================================
# 13. Plot Loss
# ==========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("CNN Training vs Validation Loss")
plt.legend()

plt.show()


# ==========================================================
# 14. Evaluate Model
# ==========================================================

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\nModel Performance")
print("-----------------")
print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")


# ==========================================================
# 15. Prediction
# ==========================================================

predictions = model.predict(X_test, verbose=0)

predicted_class = (predictions > 0.5).astype(int)

print("\nSample Prediction")
print("-----------------")
print("Predicted :", predicted_class[0][0])
print("Actual    :", y_test[0])


# ==========================================================
# 16. Save Model
# ==========================================================

model.save(MODEL_PATH)

print(f"\nModel saved successfully: {MODEL_PATH}")