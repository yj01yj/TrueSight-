# image_model.py

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import os
from model_architecture import XceptionWithCBAM

# 1. 모델 로드
model_path = os.path.join("image_model_weights", "KDF_final_model.pth")
model = XceptionWithCBAM(num_classes=2)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# 2. 이미지 전처리
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
])

# 3. 추론 함수 정의
def predict_image(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = transform(image).unsqueeze(0)  # (1, 3, 299, 299)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze().numpy()

    real_prob, fake_prob = probs[0], probs[1]
    label = "UNSURE" if abs(real_prob - fake_prob) < 0.05 else ("REAL" if real_prob > fake_prob else "FAKE")

    print("🖼️ 예측 결과:", label)
    print(f"→ REAL 확률: {real_prob:.4f}, FAKE 확률: {fake_prob:.4f}")

# 4. 테스트 실행 예시
if __name__ == "__main__":
    predict_image("test_images/sample_real_or_fake.jpg")
