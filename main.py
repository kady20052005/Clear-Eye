import io
import joblib
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("best_svm_model.pkl")

def preprocess_image(img: Image.Image):
    img = img.resize((128, 128))
    return list(img.getdata())

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    features = preprocess_image(image)
    prediction = model.predict(np.array(features).reshape(1, -1))[0]
    return {"prediction": str(prediction)}
