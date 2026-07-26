print("Loading Emotion Model...")


try:
    model = load_model(
        "saved_model/emotion_model.keras",
        compile=False
    )

    print("✅ Emotion Model Loaded")

except Exception as e:
    print("❌ Model Loading Failed")
    print(e)
    raise e
