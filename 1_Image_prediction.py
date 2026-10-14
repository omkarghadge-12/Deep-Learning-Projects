import os
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from PIL import Image


# --------------------------------------------------
# 1. Dataset Path
# --------------------------------------------------

cat_path = os.path.join("PetImages", "Cat")
dog_path = os.path.join("PetImages", "Dog")


# --------------------------------------------------
# 2. Load Images
# --------------------------------------------------

images = []
labels = []

IMG_SIZE = 64


# Cat = 0
for image_name in os.listdir(cat_path):

    image_path = os.path.join(cat_path, image_name)

    try:
        image = Image.open(image_path)
        image = image.convert("RGB")
        image = image.resize((IMG_SIZE, IMG_SIZE))

        images.append(np.array(image))
        labels.append(0)

    except:
        continue


# Dog = 1
for image_name in os.listdir(dog_path):

    image_path = os.path.join(dog_path, image_name)

    try:
        image = Image.open(image_path)
        image = image.convert("RGB")
        image = image.resize((IMG_SIZE, IMG_SIZE))

        images.append(np.array(image))
        labels.append(1)

    except:
        continue


# --------------------------------------------------
# 3. Convert into NumPy arrays
# --------------------------------------------------

X = np.array(images)
y = np.array(labels)

print("Images shape :", X.shape)
print("Labels shape :", y.shape)


# --------------------------------------------------
# 4. Normalize Images
# --------------------------------------------------

X = X.astype("float32") / 255.0


# --------------------------------------------------
# 5. Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training images :", X_train.shape)
print("Testing images  :", X_test.shape)


# --------------------------------------------------
# 6. Create CNN Model
# --------------------------------------------------

model = Sequential()

model.add(
    Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(64, 64, 3)
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


# --------------------------------------------------
# 7. Flatten + Dense Layers
# --------------------------------------------------

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


# --------------------------------------------------
# 8. Model Summary
# --------------------------------------------------

model.summary()


# --------------------------------------------------
# 9. Compile Model
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# --------------------------------------------------
# 10. Train Model
# --------------------------------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop]
)

# --------------------------------------------------
# 11. Plot Accuracy Graph
# --------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Marvellous CNN Training vs Validation Accuracy")
plt.legend()
plt.show()

# --------------------------------------------------
# 12. Plot Loss Graph
# --------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("CNN Training vs Validation Loss")
plt.legend()
plt.show()

# --------------------------------------------------
# 13. Evaluate Model
# --------------------------------------------------

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test
)

print("Test Loss :", test_loss*100)
print("Test Accuracy :", test_accuracy*100)


# --------------------------------------------------
# 14. Prediction
# -------------------------------------------------- 

predictions = model.predict(X_test)

predicted_class = (predictions > 0.5).astype(int)

print("Predicted :", predicted_class[0][0])
print("Actual    :", y_test[0])

# --------------------------------------------------
# 15. Save the model
# --------------------------------------------------
model.save("cat_dog_model.keras")