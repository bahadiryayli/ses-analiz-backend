# Sürüm: v0.2.0 | Tarih: 2026-10-31 | Değişiklik: yanıt formatı birleştirildi, temizlik
import logging
import os
import uuid

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from v0_2_0_analysis import analyze_audio

logger = logging.getLogger("ses-analiz")

app = FastAPI(title="Ses Sağlığı Analiz API")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".wav", ".m4a", ".mp3", ".aac", ".ogg"}
MAX_BYTES = 10 * 1024 * 1024  # 10 MB


def _error(status_code: int, message: str) -> JSONResponse:
    # Analiz modülüyle aynı format: status / data / error
    return JSONResponse(
        {"status": "error", "data": None, "error": message},
        status_code=status_code,
    )


API_VERSION = "0.2.0"


@app.get("/health")
def health():
    return {"status": "ok", "version": API_VERSION}


# Not: `def` (async değil) çünkü librosa analizi engelleyici bir iştir;
# FastAPI bunu ayrı bir thread'de çalıştırır.
@app.post("/upload")
def upload_audio(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return _error(400, "Desteklenmeyen dosya türü.")

    # Benzersiz isim: aynı isimli dosyalar birbirinin üstüne yazılmaz
    file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")

    try:
        size = 0
        with open(file_path, "wb") as buffer:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_BYTES:
                    return _error(413, "Dosya çok büyük (en fazla 10 MB).")
                buffer.write(chunk)

        result = analyze_audio(file_path)
        status_code = 200 if result["status"] == "success" else 422
        return JSONResponse(result, status_code=status_code)
    except Exception:
        logger.exception("Yükleme/analiz hatası")
        return _error(500, "Sunucuda beklenmeyen bir hata oluştu.")
    finally:
        # Analizden sonra dosyayı sil, sunucuda birikmesin
        if os.path.exists(file_path):
            os.remove(file_path)
