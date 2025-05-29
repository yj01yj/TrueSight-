import os
import torch
import librosa
import cv2
import numpy as np
import moviepy.editor as mp
from transformers import AutoConfig, AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
from image_model.model_architecture import XceptionWithCBAM
from torchvision import transforms

# 음성 모델 경로
AUDIO_MODEL_DIR = "audio_model/audio_model_for_git"
# 이미지 모델 경로
IMAGE_MODEL_PATH = "image_model/image_model_weights/KDF_final_model.pth"

# 🔹 0. 영상 → 음성 추출
def extract_audio(video_path, output_audio_path="temp_audio.wav"):
    clip = mp.VideoFileClip(video_path)
    clip.audio.write_audiofile(output_audio_path, fps=16000, verbose=False, logger=None)
    return output_audio_path

# 🔹 0. 영상 → 대표 프레임 추출
def extract_frame(video_path, output_frame_path="temp_frame.jpg"):
    clip = mp.VideoFileClip(video_path)
    frame = clip.get_frame(clip.duration / 2)  # 중간 지점 프레임 추출
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_frame_path, frame)
    return output_frame_path

# 1. 오디오 예측
def predict_audio(file_path):
    config = AutoConfig.from_pretrained(AUDIO_MODEL_DIR, local_files_only=True)
    extractor = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL_DIR, local_files_only=True)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(AUDIO_MODEL_DIR, config=config, local_files_only=True)

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

    print(f"🎧 음성 예측 결과: {label} (REAL: {real_prob:.4f}, FAKE: {fake_prob:.4f})")
    return fake_prob

# 2. 이미지 예측
def predict_image(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = XceptionWithCBAM()
    model.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    real_prob, fake_prob = probs[0], probs[1]
    label = "UNSURE" if abs(real_prob - fake_prob) < 0.05 else ("REAL" if real_prob > fake_prob else "FAKE")

    print(f"🖼️ 이미지 예측 결과: {label} (REAL: {real_prob:.4f}, FAKE: {fake_prob:.4f})")
    return fake_prob

# 3. Adaptive Weighted Voting
def multimodal_decision(audio_path, image_path):
    audio_fake = predict_audio(audio_path)
    image_fake = predict_image(image_path)

    if image_fake > audio_fake:
        final_score = 0.6 * image_fake + 0.4 * audio_fake
    else:
        final_score = 0.4 * image_fake + 0.6 * audio_fake

    if final_score >= 0.7:
        final = "FAKE"
    elif final_score <= 0.4:
        final = "REAL"
    else:
        final = "UNSURE"

    print(f"🧠 최종 판단 결과: {final} (통합 점수: {final_score:.4f})")
    return final

# 4. 예시 실행 (영상 하나만 입력하면 자동 분리)
if __name__ == "__main__":
    video_path = "test_samples/sample_video.mp4"

    audio_input = extract_audio(video_path)
    image_input = extract_frame(video_path)

    multimodal_decision(audio_input, image_input)

    # 사용 후 임시파일 삭제 (선택사항)
    os.remove(audio_input)
    os.remove(image_input)
