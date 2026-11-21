import io
import joblib
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
app = FastAPI()

# السماح للواجهة بالاتصال بالخادم
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# تحميل نموذج SVM المحفوظ
model = joblib.load("best_svm_model.pkl")


def preprocess_image(img: Image.Image):
    # مطابقة خطوات المعالجة المستخدمة في التدريب
    img = img.resize((128, 128))
    return list(img.getdata())


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    features = preprocess_image(image)

    # التنبؤ بالفئة
prediction = model.predict(np.array(features).reshape(1,-1))[O] 
return {"prediction": str(prediction)}
