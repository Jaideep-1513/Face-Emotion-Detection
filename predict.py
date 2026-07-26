import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# ==========================================
# LOAD TRAINED MODEL
# ==========================================

print("Loading trained model...")

model = load_model("saved_model/emotion_model.keras")

print("✅ Model Loaded Successfully!")

# ==========================================
# EMOTION LABELS
# ==========================================

emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# ==========================================
# OPEN FILE DIALOG
# ==========================================

root = tk.Tk()
root.withdraw()

print("\nPlease select an image...")

image_path = filedialog.askopenfilename(
    title="Select a Face Image",
    filetypes=[
        ("Image Files", "*.jpg *.jpeg *.png *.bmp")
    ]
)

if not image_path:
    print("❌ No image selected.")
    exit()

print(f"Selected Image: {image_path}")

# ==========================================
# LOAD IMAGE
# ==========================================

img = Image.open(image_path).convert("L")
display_img = img.copy()

img = img.resize((48, 48))

img_array = np.array(img)

img_array = img_array.astype("float32") / 255.0

img_array = img_array.reshape(1, 48, 48, 1)

# ==========================================
# PREDICT
# ==========================================

prediction = model.predict(img_array, verbose=0)

predicted_index = np.argmax(prediction)

emotion = emotion_labels[predicted_index]

confidence = prediction[0][predicted_index] * 100

# ==========================================
# DISPLAY RESULT
# ==========================================

plt.figure(figsize=(6, 6))
plt.imshow(display_img, cmap="gray")
plt.title(f"{emotion} ({confidence:.2f}%)", fontsize=16)
plt.axis("off")
plt.show()

# ==========================================
# PRINT RESULTS
# ==========================================

print("\n==============================")
print("PREDICTION RESULT")
print("==============================")

print(f"Predicted Emotion : {emotion}")
print(f"Confidence        : {confidence:.2f}%")

print("\nAll Emotion Probabilities:")

for i, label in enumerate(emotion_labels):
    print(f"{label:10}: {prediction[0][i] * 100:.2f}%")
