# app.py
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

st.set_page_config(page_title="산타 타격 횟수 분석기", layout="centered")
st.title("산타 타격 횟수 분석기")

st.markdown("""
이 웹앱은 산타 경기 영상에서 손과 발의 움직임을 추적하여
파란 선수와 빨간 선수 각각의 펀치(손), 킥(발) 횟수를 자동으로 추정하고,
타격 타이밍을 시각화하며 분석 결과를 CSV와 그래프로 제공합니다.
""")

uploaded_file = st.file_uploader("분석할 영상 파일(mp4)을 업로드하세요", type=["mp4", "avi"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(uploaded_file)
    st.write("영상 분석 중...")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    blue_punch_count = 0
    red_punch_count = 0
    blue_kick_count = 0
    red_kick_count = 0

    prev_blue_hand = None
    prev_red_hand = None
    prev_blue_foot = None
    prev_red_foot = None

    movement_threshold = 30
    frame_num = 0
    events = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        frame_center = frame.shape[1] // 2

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            left_hand = (int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * frame.shape[1]),
                         int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * frame.shape[0]))
            right_hand = (int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * frame.shape[1]),
                          int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * frame.shape[0]))
            left_foot = (int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x * frame.shape[1]),
                         int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * frame.shape[0]))
            right_foot = (int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x * frame.shape[1]),
                          int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * frame.shape[0]))

            timestamp = str(timedelta(seconds=frame_num / fps))

            for hand, prev_hand, label in [(left_hand, prev_blue_hand, 'blue_punch'), (right_hand, prev_red_hand, 'red_punch')]:
                if prev_hand and np.linalg.norm(np.array(hand) - np.array(prev_hand)) > movement_threshold:
                    if hand[0] < frame_center:
                        blue_punch_count += 1
                        events.append((timestamp, "파란 선수", "펀치"))
                    else:
                        red_punch_count += 1
                        events.append((timestamp, "빨간 선수", "펀치"))
                if hand[0] < frame_center:
                    prev_blue_hand = hand
                else:
                    prev_red_hand = hand

            for foot, prev_foot, label in [(left_foot, prev_blue_foot, 'blue_kick'), (right_foot, prev_red_foot, 'red_kick')]:
                if prev_foot and np.linalg.norm(np.array(foot) - np.array(prev_foot)) > movement_threshold:
                    if foot[0] < frame_center:
                        blue_kick_count += 1
                        events.append((timestamp, "파란 선수", "킥"))
                    else:
                        red_kick_count += 1
                        events.append((timestamp, "빨간 선수", "킥"))
                if foot[0] < frame_center:
                    prev_blue_foot = foot
                else:
                    prev_red_foot = foot

    cap.release()

    st.success(f"[파란 선수] 펀치: {blue_punch_count}회 | 킥: {blue_kick_count}회")
    st.success(f"[빨간 선수] 펀치: {red_punch_count}회 | 킥: {red_kick_count}회")

    df = pd.DataFrame(events, columns=["시간", "선수", "기술"])
    st.subheader("타격/킥 시점 로그")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="CSV로 결과 저장",
        data=csv,
        file_name="타격_킥_분석결과.csv",
        mime="text/csv"
    )

    # 그래프 시각화
    st.subheader("선수별 기술 사용 통계 그래프")
    if not df.empty:
        chart_df = df.groupby(["선수", "기술"]).size().unstack(fill_value=0)
        st.bar_chart(chart_df)
    else:
        st.info("그래프를 표시할 데이터가 없습니다.")
