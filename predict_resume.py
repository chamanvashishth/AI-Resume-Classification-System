import joblib

model = joblib.load("resume_classifier.joblib")

resume = input("Paste resume text:\n")

prediction = model.predict([resume])[0]
confidence = model.predict_proba([resume]).max()

print("\nPredicted Category:", prediction)
print("Confidence:", f"{confidence:.2%}")
