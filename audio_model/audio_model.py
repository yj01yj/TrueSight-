# audio_model.py

import os
import torch
import librosa
from transformers import AutoConfig, AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

def load_audio_model(model_dir):
    """
    지정한 모델 경로에서 구성(config), extractor, 모델 가중치를 로드합니다.
    """
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    extractor = AutoFeatureExtractor.from_pretrained(model_dir, local_files_only=True)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_dir, config=config, local_files_only=True
    )
    return model, extractor

def predict_audio(file_path, model_dir):
    """
    주어진 오디오 파일의 딥페이크 여부를 예측합니다.

    Args:
        file_path (str): 오디오 파일 경로 (.wav 형식)
        model_dir (str): 저장된 모델 파일들이 있는 폴더 경로

    Prints:
        예측 결과와 확률 정보
    """
    model, extractor = load_audio_model(model_dir)

    speech, _ = librosa.load(file_path, sr=16000)
    if len(speech) > 16000 * 15:
        speech = speech[:16000 * 15]

    inputs = extractor(speech, sampling_rate=16000, return_tensors="pt", padding=True)

    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    real_prob, fake_prob = probs[0], probs[1]
    label = "UNSURE" if abs(real_prob - fake_prob) < 0.05 else ("REAL" if real_prob > fake_prob else "FAKE")

    print("🎧 예측 결과:", label)
    print(f"→ REAL 확률: {real_prob:.4f}, FAKE 확률: {fake_prob:.4f}")

# 사용 예시 (직접 실행 시)
if __name__ == "__main__":
    predict_audio("your_audio.wav", model_dir="audio_model_for_git")
