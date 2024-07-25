import tensorflow as tf
import cv2
import numpy as np

# 미리 학습된 모델 로드 (모델 파일 경로는 실제 모델 파일로 대체해야 함)
model = tf.keras.models.load_model('runs\\detect\\train4\\weights\\best.pt')

# 웹캠 초기화
cap = cv2.VideoCapture(0)

while True:
    # 웹캠에서 프레임 읽기
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # 프레임 전처리 (모델 입력 형식에 맞게 조정)
    img = cv2.resize(frame, (224, 224))  # 모델 입력 크기에 맞게 조정
    img = img.astype('float32') / 255.0  # 스케일링
    img = np.expand_dims(img, axis=0)  # 배치 차원 추가

    # 손 모양 감지
    predictions = model.predict(img)
    label = np.argmax(predictions, axis=1)
    
    # 결과 표시
    cv2.putText(frame, f'Label: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Hand Gesture Recognition', frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 및 윈도우 정리
cap.release()
cv2.destroyAllWindows()
