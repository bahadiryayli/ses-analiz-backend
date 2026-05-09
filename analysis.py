import librosa
import numpy as np

def analyze_audio(file_path: str):
    try:
        # Ses dosyasını yükle
        y, sr = librosa.load(file_path, sr=16000,)

        fmin = 80
        fmax = 400
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=fmin, fmax=fmax) 
        # sessiz kısımları kırpııyoruz
        y_trimmed, _ = librosa.effects.trim(y)

        # Süre
        duration = librosa.get_duration(y=y, sr=sr)

        # RMS Energy
        rms = np.mean(librosa.feature.rms(y=y_trimmed))

        # Zero Crossing Rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(y_trimmed))

        # Pitch (Fundamental Frequency)
        pitch_values = f0[~np.isnan(f0)]
        pitch_mean = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0

        risk_notu = "normal"
        if pitch_mean > 0 and (pitch_mean < 80 or pitch_mean > 250):
            risk_notu = "Ses frekansi alişilmişin dişinda. Bir uzmana danişilabilir."
        # MFCC (ilk 13)
        mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1).tolist()

        return {
            "status": "success",
            "data": {
                "pitch_hz": round(pitch_mean, 2),
                "energy": round(float(rms), 4),
                "zcr": round(float(zcr),4),
                "mfcc_mean": mfcc_mean,
                "duration_sec": round(librosa.get_duration(y=y_trimmed, sr=sr), 2),
                "interpretation": risk_notu    
            }
        }
    except Exception as e:
        return {"status": "error", "massage": str(e)}