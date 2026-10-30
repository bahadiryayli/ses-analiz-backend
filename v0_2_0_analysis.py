# Sürüm: v0.2.0 | Tarih: 2026-10-31 | Değişiklik: yanıt formatı birleştirildi, temizlik
import librosa
import numpy as np

SR = 16000
MIN_DURATION_SEC = 1.5
# Çocuk, kadın ve erkek seslerini kapsayan geniş aralık
FMIN_HZ = 65
FMAX_HZ = 500

DISCLAIMER = (
    "Bu sonuçlar tıbbi tanı değildir; yalnızca ses özelliklerinin "
    "ölçümüdür. Ses değişiklikleri devam ederse bir sağlık "
    "uzmanına danışın."
)


def _ok(data: dict) -> dict:
    return {"status": "success", "data": data, "error": None}


def _fail(message: str) -> dict:
    return {"status": "error", "data": None, "error": message}


def _build_notes(duration: float, voiced_ratio: float) -> list:
    notes = []
    if voiced_ratio < 0.3:
        notes.append(
            "Kaydın büyük kısmında belirgin ses algılanamadı. Sessiz bir "
            "ortamda, telefonu ağzınıza yakın tutarak birkaç saniye "
            "'aaa' sesi çıkarıp tekrar deneyin."
        )
    if duration < 3:
        notes.append(
            "Kayıt kısa. Daha güvenilir ölçüm için en az 3-5 saniye kayıt alın."
        )
    if not notes:
        notes.append(
            "Ölçümler başarıyla alındı. Zaman içinde kendi kayıtlarınızla "
            "karşılaştırmak, tek bir kayda bakmaktan daha anlamlıdır."
        )
    return notes


def analyze_audio(file_path: str) -> dict:
    try:
        y, sr = librosa.load(file_path, sr=SR, mono=True)
        y_trimmed, _ = librosa.effects.trim(y, top_db=30)

        duration = len(y_trimmed) / sr
        if duration < MIN_DURATION_SEC:
            return _fail(
                "Kayıt çok kısa veya çok sessiz. Lütfen en az 2 saniye, "
                "net bir ses kaydedin."
            )

        rms = float(np.mean(librosa.feature.rms(y=y_trimmed)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y_trimmed)))

        f0, _, _ = librosa.pyin(y_trimmed, fmin=FMIN_HZ, fmax=FMAX_HZ, sr=sr)
        pitch_values = f0[~np.isnan(f0)]
        if len(pitch_values) == 0:
            return _fail(
                "Kayıtta ses perdesi algılanamadı. Sessiz bir ortamda "
                "'aaa' sesi çıkararak tekrar deneyin."
            )

        voiced_ratio = float(len(pitch_values) / len(f0))
        pitch_mean = float(np.mean(pitch_values))
        pitch_std = float(np.std(pitch_values))

        mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13)
        mfcc_mean = [round(float(v), 3) for v in np.mean(mfcc, axis=1)]

        return _ok(
            {
                "pitch_mean_hz": round(pitch_mean, 2),
                "pitch_std_hz": round(pitch_std, 2),
                "energy": round(rms, 4),
                "zcr": round(zcr, 4),
                "voiced_ratio": round(voiced_ratio, 2),
                "duration_sec": round(duration, 2),
                "mfcc_mean": mfcc_mean,
                "notes": _build_notes(duration, voiced_ratio),
                "disclaimer": DISCLAIMER,
            }
        )
    except Exception as e:
        return _fail(f"Ses dosyası işlenemedi: {e}")
