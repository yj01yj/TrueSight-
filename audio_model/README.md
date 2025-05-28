# 🎧 audio_model - Audio Deepfake Detection

이 모듈은 Wav2Vec2 기반 딥러닝 모델을 사용하여 음성 파일이 **진짜(REAL)**인지 **딥페이크(FAKE)**인지 판별합니다.  
`audio_model.py`를 실행하면 `.wav` 오디오 파일의 딥페이크 여부를 예측할 수 있습니다.

---

## 📁 폴더 구성
```
audio_model/
├── audio_model.py                # 딥페이크 추론 코드 (범용 구조)
├── README.md                     # 음성 모델에 대한 설명
└── audio_model_for_git/          # (사용자가 직접 추가해야 하는 모델 폴더)
    ├── config.json
    ├── pytorch_model.bin
    └── preprocessor_config.json
```

---

## 📥 모델 다운로드

🔗 [Google Drive 모델 다운로드 링크](https://drive.google.com/drive/folders/1-5bpVkglTfsdJB_9BixgS8kY-JGBAhRh?usp=sharing)

> 다운로드한 후, `audio_model/` 폴더 안에 `audio_model_for_git/` 폴더 전체를 그대로 넣어주세요.

---

## 📦 설치 필요 패키지

```bash
pip install transformers torch librosa
