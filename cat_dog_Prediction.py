import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

model = load_model("cat_dog_model.keras")

image = Image.open("mycat.jpg")
image = image.convert("RGB")
image = image.resize((64, 64))

image = np.array(image)
image = image.astype("float32") / 255.0

image = np.expand_dims(image, axis=0)

prediction = model.predict(image)

if prediction[0][0] > 0.5:
    print("Prediction: Dog")
else:
    print("Prediction: Cat")