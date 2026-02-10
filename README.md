# 🧠 TrueSight - Multimodal Deepfake Detection

이 프로젝트는 이미지와 음성 데이터를 함께 사용하여 딥페이크(Deepfake) 여부를 판단하는 **멀티모달 기반 딥페이크 탐지 시스템**입니다.

음성 모델(Wav2Vec2), 이미지 모델(Xception + CBAM)을 통해 정밀한 판별을 수행하며,`multimodal_detector.py`를 통해 두 결과를 종합하여 최종 판단을 내립니다.

---


최근 딥페이크 기술이 빠르게 발전하면서, 사람의 얼굴을 조작하거나 음성을 위조하는 사례가 증가하고 있습니다.  
특히 이미지나 음성만을 이용한 탐지 방식은 각각의 한계가 존재하기 때문에, 이 프로젝트에서는 두 가지 정보를 **동시에 분석**하여 보다 정확한 탐지를 수행하고자 하였습니다.

---

## 🎯 프로젝트 목표

- 영상에서 얼굴 이미지와 음성 데이터를 추출한 뒤,
- 각각의 딥러닝 모델로 분석하여
- 두 결과를 종합해 딥페이크 여부를 최종 판단하는 **멀티모달 탐지 모델**을 구현하는 것이 목표입니다.

---

## 🧠 모델 구성

### 1. 이미지 기반 탐지
- **사용 모델**: Xception 구조에 CBAM(주의 메커니즘)을 결합한 `XceptionWithCBAM`
- **기능**: 영상 속 얼굴의 미세한 특징을 분석하여 딥페이크 여부를 판단

### 2. 음성 기반 탐지
- **사용 모델**: `facebook/wav2vec2-large-xlsr-53` (HuggingFace 제공)
- **기능**: 사람 목소리의 음성 패턴을 분석하여 TTS(가짜 음성) 여부를 판별

### 3. 최종 통합
- 이미지 모델과 음성 모델의 예측 결과를 비교한 후, **다수결 방식**으로 최종 판단

---

## 📁 폴더 구조
```
TrueSight/
├── audio_model/                  # 음성 기반 딥페이크 탐지 모듈
│   ├── audio_model.py
│   ├── README.md
│   └── audio_model_for_git/      # (사용자가 직접 추가해야 하는 모델 폴더)🔗 Drive로 공유
│       └── ...
│
├── image_model/                  # 이미지 기반 딥페이크 탐지 모듈
│   ├── image_model.py
│   ├── model_architecture.py
│   ├── README.md
│   └── image_model_weights/      # (사용자가 직접 추가해야 하는 모델 폴더)🔗 Drive로 공유
│       └── KDF_final_model.pth
│
├── multimodal_detector.py        # ✅ 이미지 + 음성 통합 판단 스크립트
├── requirements.txt              # 공통 라이브러리 목록
└── README.md                     # (이 문서)
```

---

## ⚙️ 시스템 작동 방식

1. 사용자가 영상을 업로드하면,
2. 영상에서 얼굴 이미지와 음성을 자동으로 분리합니다.
3. 각각의 데이터를 모델에 입력하여 예측 결과를 얻고,
4. 결과를 종합하여 웹사이트에 딥페이크 여부를 출력합니다.

---

실행 결과 예시:
```
🎧 음성 예측 결과: REAL (REAL: 0.9613, FAKE: 0.0387)
🖼️ 이미지 예측 결과: FAKE (REAL: 0.1042, FAKE: 0.8958)
🧠 최종 판단 결과: FAKE (통합 점수: 0.5930 → 0.6 * 이미지 + 0.4 * 음성)
```

---

## 💡 통합 판단 기준

- 이미지 모델과 음성 모델의 **FAKE 확률 중 더 높은 쪽에 더 많은 가중치(0.6)** 를 부여하고
다른 쪽에는 0.4를 곱해 최종 통합 점수를 계산합니다.
- 이 통합 점수가 0.7 이상이면 FAKE, 0.4 이하이면 REAL, 그 사이면 UNSURE로 판단합니다.
- 이 방식은 두 모달리티의 정보를 동적으로 통합하며, 조작된 모달리티의 영향을 더 반영할 수 있는 **Adaptive Weighted Voting** 방식입니다.

---

## 📌 참고
- `audio_model/` 및 `image_model/` 각각의 폴더 안에 README가 포함되어 있어요.
- `multimodal_detector.py`는 두 모델을 호출해 결과를 종합하는 통합 실행 진입점입니다.

---

## 🌐 구현 환경

- **Colab**: 모델 학습 및 API 서버 실행
- **Replit**: 웹사이트 UI 구현 (Flask 프레임워크)
- **localtunnel**: Colab 서버를 외부에 연결하여 Replit과 연동

---

## 📊 성능 평가 지표

- **FID** (Fréchet Inception Distance)
- **SSIM** (Structural Similarity)
- **PSNR** (Peak Signal-to-Noise Ratio)
- **IS** (Inception Score)
- **LPIPS** (Perceptual Image Patch Similarity)

각 모델의 성능을 위 지표로 정량적으로 평가하였습니다.

---

## 🧩 사용 기술

- Python
- PyTorch
- HuggingFace Transformers
- Flask
- moviepy, librosa (영상 및 음성 처리용)
- Google Colab, Replit, localtunnel

---

## 👩‍💻 프로젝트 기여자

- 심유정 : 데이터 수집 및 모델 구축
- 백승호 : 모델 통합 및 웹사이트 구축
- 송주환 : 데이터 수집 및 모델 구축

---


