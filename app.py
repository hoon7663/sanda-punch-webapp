import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import time
import math

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

st.set_page_config(page_title="산타 타격 횟수 분석기", layout="centered")
st.title("산타 타격 횟수 분석기")

st.markdown("""
🎯 정밀도 최상위 버전입니다.  
- 수직 이동 제외  
- 일정 속도 이상만 타격 인식  
- 관절 꺾임 각도 변화 기반 동작 분석 강화
""")

uploaded_file = st.file_uploader("분석할 영상 파일(mp4)을 업로드하세요", type=["mp4", "avi"])

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

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

    prev_left_arm_angle = None
    prev_right_arm_angle = None
    prev_left_leg_angle = None
    prev_right_leg_angle = None

    last_blue_punch_time = 0
    last_red_punch_time = 0
    last_blue_kick_time = 0
    last_red_kick_time = 0

    movement_threshold = 60
    cooldown = 0.5
    min_speed = 100
    min_angle_change = 15  # 관절 각도 변화 기준

    frame_num = 0
    events = []

    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        if frame_num % 3 != 0:
            continue

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        frame_center = frame.shape[1] // 2
        timestamp = str(timedelta(seconds=frame_num / fps))

        now = time.time()
        time_diff = now - prev_time
        prev_time = now

        if results.pose_landmarks:
            lms = results.pose_landmarks.landmark

            lh = (int(lms[mp_pose.PoseLandmark.LEFT_WRIST].x * frame.shape[1]), int(lms[mp_pose.PoseLandmark.LEFT_WRIST].y * frame.shape[0]))
            rh = (int(lms[mp_pose.PoseLandmark.RIGHT_WRIST].x * frame.shape[1]), int(lms[mp_pose.PoseLandmark.RIGHT_WRIST].y * frame.shape[0]))
            la = (int(lms[mp_pose.PoseLandmark.LEFT_ANKLE].x * frame.shape[1]), int(lms[mp_pose.PoseLandmark.LEFT_ANKLE].y * frame.shape[0]))
            ra = (int(lms[mp_pose.PoseLandmark.RIGHT_ANKLE].x * frame.shape[1]), int(lms[mp_pose.PoseLandmark.RIGHT_ANKLE].y * frame.shape[0]))

            lel = (int(lms[mp_pose.PoseLandmark.LEFT_ELBOW].x * frame.shape[1]), int(lms[mp_pose.PoseLandmark.LEFT_ELBOW].y * frame.shape[0]))
            rel = (int(lms[mp_pose.PoseLandmark.RIGHT_ELBOW].x * frame.shape[1]), int(lms[mp_pose.PoseLandmark.RIGHT_ELBOW].y * frame.shape[0]))
            lsh = (int(lms[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame.shape[1]), int(lms[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame.shape[0]))
            rsh = (int(lms[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame.shape[1]), int(lms[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame.shape[0]))

            lkn = (int(lms[mp_pose.PoseLandmark.LEFT_KNEE].x * frame.shape[1]), int(lms[mp_pose.PoseLandmark.LEFT_KNEE].y * frame.shape[0]))
            rkn = (int(lms[mp_pose.PoseLandmark.RIGHT_KNEE].x * frame.shape[1]), int(lms[mp_pose.PoseLandmark.RIGHT_KNEE].y * frame.shape[0]))
            lhip = (int(lms[mp_pose.PoseLandmark.LEFT_HIP].x * frame.shape[1]), int(lms[mp_pose.PoseLandmark.LEFT_HIP].y * frame.shape[0]))
            rhip = (int(lms[mp_pose.PoseLandmark.RIGHT_HIP].x * frame.shape[1]), int(lms[mp_pose.PoseLandmark.RIGHT_HIP].y * frame.shape[0]))

            left_arm_angle = calculate_angle(lsh, lel, lh)
            right_arm_angle = calculate_angle(rsh, rel, rh)
            left_leg_angle = calculate_angle(lhip, lkn, la)
            right_leg_angle = calculate_angle(rhip, rkn, ra)

            def movement_valid(curr, prev):
                dist = np.linalg.norm(np.array(curr) - np.array(prev))
                return dist > movement_threshold and dist / max(time_diff, 0.01) > min_speed

            if (
                prev_left_arm_angle and abs(left_arm_angle - prev_left_arm_angle) > min_angle_change and
                prev_blue_hand and movement_valid(lh, prev_blue_hand) and lh[0] < frame_center and now - last_blue_punch_time > cooldown
            ):
                blue_punch_count += 1
                last_blue_punch_time = now
                events.append((timestamp, "파란 선수", "펀치"))

            if (
                prev_right_arm_angle and abs(right_arm_angle - prev_right_arm_angle) > min_angle_change and
                prev_red_hand and movement_valid(rh, prev_red_hand) and rh[0] >= frame_center and now - last_red_punch_time > cooldown
            ):
                red_punch_count += 1
                last_red_punch_time = now
                events.append((timestamp, "빨간 선수", "펀치"))

            if (
                prev_left_leg_angle and abs(left_leg_angle - prev_left_leg_angle) > min_angle_change and
                prev_blue_foot and movement_valid(la, prev_blue_foot) and la[0] < frame_center and now - last_blue_kick_time > cooldown
            ):
                blue_kick_count += 1
                last_blue_kick_time = now
                events.append((timestamp, "파란 선수", "킥"))

            if (
                prev_right_leg_angle and abs(right_leg_angle - prev_right_leg_angle) > min_angle_change and
                prev_red_foot and movement_valid(ra, prev_red_foot) and ra[0] >= frame_center and now - last_red_kick_time > cooldown
            ):
                red_kick_count += 1
                last_red_kick_time = now
                events.append((timestamp, "빨간 선수", "킥"))

            prev_blue_hand = lh
            prev_red_hand = rh
            prev_blue_foot = la
            prev_red_foot = ra

            prev_left_arm_angle = left_arm_angle
            prev_right_arm_angle = right_arm_angle
            prev_left_leg_angle = left_leg_angle
            prev_right_leg_angle = right_leg_angle

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

    st.subheader("선수별 기술 사용 통계 그래프")
    if not df.empty:
        chart_df = df.groupby(["선수", "기술"]).size().unstack(fill_value=0)
        st.bar_chart(chart_df)
    else:
        st.info("그래프를 표시할 데이터가 없습니다.")
