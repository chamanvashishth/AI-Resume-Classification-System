from model_utils import load_or_train_model

model = load_or_train_model()

resume = input("Paste resume text:\n").strip()

if not resume:
    raise ValueError("Resume text cannot be empty.")

prediction = model.predict([resume])[0]
confidence = model.predict_proba([resume]).max()

print("\nPredicted Category:", prediction)
print("Confidence:", f"{confidence:.2%}")
