import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2
import numpy as np
import os, re
import pandas as pd

# K-NN 모델 로드 및 학습
def load_gesture_model(csv_path="data_hand.csv"):
    try:
        data = np.genfromtxt(csv_path, delimiter=',')

        if data.ndim == 1:
            # 데이터 행이 1개일 경우 reshape (1, 16)
            data = data.reshape(1, -1)

        angles = data[:, :-1].astype(np.float32) # 각도 데이터(특징)
        labels = data[:, -1].astype(np.int32) # 라벨 데이터

        GLOBAL_TRAIN_ANGLES = angles
        GLOBAL_TRAIN_LABELS = labels

        if angles.shape[1] != 1:
            print(f"오류: CSV 파일의 특징 개수가 1개가 아닙니다. 실제 개수: {angles.shape[1]}")
            return None
    
    except Exception as e:
        print(f"오류: '{csv_path}' 파일을 로드하거나 처리하는 중 문제가 발생했습니다: {e}")
        print("data_collector.py를 실행하여 data_hand.csv 파일을 먼저 생성해야 합니다.")
        return None
    
    # k-NN 모델 초기화 및 학습
    knn = cv2.ml.KNearest_create()
    knn.train(angles, cv2.ml.ROW_SAMPLE, labels)
    print(f"k-NN 모델 학습 완료. 총 {len(labels)}개의 데이터 사용")
    return knn

# MediaPipe 설정
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
# max_num_hands=1로 설정하여 하나의 손만 인식합니다.
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence = 0.5)
GLOBAL_TRAIN_ANGLES = pd.read_csv('data_hand.csv', header=None).values

GESTURE_LABELS = {
    1: 'OKAY', 2: 'PINCH', 3:'PEACE', 4: 'LOVE', 5: 'CALL', 6: 'POINTUP', 
    7: 'YOU', 8:'GOOD', 9: 'BAD', 10: 'FIST'
}

# 모양 특징 추출 함수: 관절 각도 계산
def calculate_joint_angles(joint):
    v1_indices = [0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
    v2_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    v1 = joint[v1_indices, :]
    v2 = joint[v2_indices, :]

    v = v2 - v1 # 20개의 뼈대 벡터 (shape: (20, 3))

    # 벡터 정규화 (크기를 1로 만듦)
    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

    # 관절 각도 15개를 계산하기 위해 인접한 벡터를 선택하는 인덱스 (길이 15)
    angle_v1_indices = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18]
    angle_v2_indices = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
    
    # 내적 후 arccos으로 각도 계산 (15개 관절 각도)
    angle = np.arccos(np.einsum('nt,nt->n',
        v[angle_v1_indices, :],
        v[angle_v2_indices, :]
    ))
    angle = np.degrees(angle)
    return angle


# 제스처 인식 및 유사도 판별
def recognize_hand_gesture(img, emoji):
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    # 예측 결과를 이미지에 시각화하기 위한 복사본
    annotated_img = img.copy()
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 랜드마크 시각화 (빨간 점과 연결선)
            mp_drawing.draw_landmarks(
                annotated_img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                # 랜드마크 스타일 설정
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2), # Green landmarks
                # 연결선 스타일 설정
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2) # Blue connections
            )
            # 랜드마크 좌표 추출
            joint = np.zeros((21, 3))
            for j, lm in enumerate(hand_landmarks.landmark):
                joint[j] = [lm.x, lm.y, lm.z]
            # 관절 각도 계산 (15개 특징 벡터)
            current_angles = calculate_joint_angles(joint)
            label = int(emoji.split('_')[0])
            train_angles = GLOBAL_TRAIN_ANGLES[GLOBAL_TRAIN_ANGLES[:,15] == label][0, :15]
            distance = np.linalg.norm(current_angles - train_angles)
            sigma = 100
            similarity = np.exp(-(distance ** 2) / (2 * sigma ** 2))
            print(similarity * 100)
            return similarity * 100
    # 손이 인식되지 않았을 경우
    return 0

if __name__ == "__main__":
        
    TEST_IMAGE_PATH = './test_yr/fist.jpg' 
    
    test_img = cv2.imread(TEST_IMAGE_PATH)
    
    if test_img is None:
        print(f"오류: 테스트 이미지 파일을 찾을 수 없습니다: '{TEST_IMAGE_PATH}'")
        print("파일 경로를 확인하거나, test_image.jpg 파일을 준비하세요.")
    else:
        # 제스처 인식 실행
        gesture_id, annotated_img, distance = recognize_hand_gesture(test_img)
        
        if gesture_id is not None:
            # 예측된 라벨 출력
            predicted_label = GESTURE_LABELS.get(gesture_id, f'Unknown ID:{gesture_id}')
            print(f'=====================================================')
            print(f'인식된 제스처 ID: {gesture_id} ({predicted_label})')
            print(f'유사도 (최소 거리): {distance:.2f}')
            print(f'=====================================================')
        else:
            print('손 인식 실패 또는 data_hand.csv에 유효한 데이터가 없습니다.')

        # 결과 이미지 표시
        cv2.imshow('Hand Gesture Recognition Result', annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # cap = cv2.VideoCapture(0)
    # if not cap.isOpened():
    #     exit()
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     cv2.imshow('frame', frame)
    #     p = recognize_hand_gesture(frame)
    #     print(p)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cap.release()
    # cv2.destroyAllWindows()