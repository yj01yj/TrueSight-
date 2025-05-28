# 📷 image_model - Image Deepfake Detection

이 모듈은 Xception + CBAM 기반 딥러닝 모델을 사용하여 이미지가 **진짜(REAL)**인지 **딥페이크(FAKE)**인지 판별합니다.  
`image_model.py`를 실행하면 `.jpg` 혹은 `.png` 이미지의 딥페이크 여부를 예측할 수 있습니다.

---

## 📁 폴더 구성
```
image_model/
├── image_model.py             # 이미지 딥페이크 추론 코드
├── model_architecture.py     # Xception + CBAM 모델 구조 정의
├── README.md                 # 이미지 모델에 대한 설명
└── image_model_weights/      # (사용자가 직접 추가해야 하는 모델 폴더)
    └── KDF_final_model.pth   # 학습된 모델 가중치 파일
```

---

## 📥 모델 다운로드

🔗 [Google Drive 모델 다운로드 링크](https://drive.google.com/file/d/1UdDOJ3XYlK_G1nMlsxGWqf0PYSU0IxYD/view?usp=sharing)

> 다운로드한 후, `image_model/` 폴더 안에 `image_model_weights/` 폴더를 만들고 그 안에 `KDF_final_model.pth` 파일을 넣어주세요.

---

## 📦 설치 필요 패키지
```bash
pip install torch torchvision opencv-python
```

---

## 🧪 사용 예시

```bash
# 터미널에서 실행하거나
python image_model.py

# 또는 image_model.py 내부
predict_image("test_images/sample_real_or_fake.jpg")
```

예측 결과는 다음과 같이 출력됩니다:
```
🖼️ 예측 결과: REAL
→ REAL 확률: 0.9823, FAKE 확률: 0.0177
```
