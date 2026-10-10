from PIL import Image

import numpy as np

img = Image.open("Digit_28x28.png")

img = img.convert("L")

img = img.resize((28,28))

pixels = np.array(img)

print("Image size : ",pixels.shape)

print("Pixel Values : ")
print(pixels)

# 0     Pure Black
# 255   Pure White
# 50    Dark grey
# 120   Medium grey
# 200   Light grey