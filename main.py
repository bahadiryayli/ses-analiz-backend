from fastapi import FastAPI, UploadFile, File
import shutil
import os
from analysis import analyze_audio

from analysis import analyze_audio

app = FastAPI(title="ISDDDA Backend")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# 1️⃣ Ses yükleme endpoint'i
@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    # 1. adım dosya kaydet
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 2. adım analiz
        analysis_result = analyze_audio(file_path)
        # başarılıysa gönder
        return analysis_result
    except Exception as e:
        # hata varsa mesaj gönder
        return {
            "status": "error",
            "data": {
                "pitch_hz": 0,
                "interpretation": f"Analiz hatası: {str(e)}"
            }
        }

# 2️⃣ Analiz endpoint'i
@app.get("/analyze/{filename}")
def analyze(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(file_path):
        return {"error": "Dosya bulunamadı"}

    result = analyze_audio(file_path)
    return result
