from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D , Dense , MaxPool2D , Flatten

model = Sequential([
    Conv2D(32, (2,2), activation='relu'),
    MaxPool2D((2,2)),
    Flatten(),
    Dense(128 , activation='relu'),
    Dense(10 , activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
