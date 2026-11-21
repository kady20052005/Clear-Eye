import io
import joblib
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from skimage.feature import hog

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

model = joblib.load("best_svm_model.pkl")

categories = ['Eye inflammation', 'Healthy eye', 'Hordeolum']

    return features.reshape(1, -1)

@app.post("/predict")
async def predict(file: UploadFile):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        features = preprocess_image(image)
        prediction = model.predict(features)[0]
        return {"prediction": str(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
