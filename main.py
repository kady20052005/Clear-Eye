import io
import joblib
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from skimage.feature import hog

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("best_svm_model.pkl")

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("L").resize((128, 128))
    img_array = np.array(image)
    features = hog(img_array, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    return features.reshape(1, -1)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        features = preprocess_image(image)
        prediction = model.predict(features)[0]
        return {"prediction": str(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
