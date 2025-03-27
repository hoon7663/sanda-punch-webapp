import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import tempfile

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

st.set_page_config(page_title="산타 타격 횟수 분석기", layout="centered")
st.title("산타 타격 횟수 분석기")

st.markdown("""
이 웹앱은 산타 경기 영상에서 손의 움직임을 추적하여
타격 횟수를 자동으로 추정합니다. 영상을 업로드하고 결과를 확인하세요.
""")

uploaded_file = st.file_uploader("분석할 영상 파일(mp4)을 업로드하세요", type=["mp4", "avi"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(uploaded_file)
    st.write("영상 분석 중...")

    cap = cv2.VideoCapture(video_path)
    punch_count = 0
    prev_left_hand = None
    prev_right_hand = None
    movement_threshold = 30

    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_hand = (int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * frame.shape[1]),
                         int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * frame.shape[0]))
            right_hand = (int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * frame.shape[1]),
                          int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * frame.shape[0]))

            if prev_left_hand and np.linalg.norm(np.array(left_hand) - np.array(prev_left_hand)) > movement_threshold:
                punch_count += 1

            if prev_right_hand and np.linalg.norm(np.array(right_hand) - np.array(prev_right_hand)) > movement_threshold:
                punch_count += 1

            prev_left_hand = left_hand
            prev_right_hand = right_hand

    cap.release()
    st.success(f"예상 타격 횟수: {punch_count}회")
