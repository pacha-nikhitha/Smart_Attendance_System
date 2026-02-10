import streamlit as st
import cv2
import face_recognition
import numpy as np
import pandas as pd
import os
from datetime import datetime

# --- CONFIGURATION ---
FACES_DIR = "faces"
ATTENDANCE_FILE = "attendance.csv"

if not os.path.exists(FACES_DIR):
    os.makedirs(FACES_DIR)

if not os.path.exists(ATTENDANCE_FILE):
    df = pd.DataFrame(columns=["Name", "Date", "Time"])
    df.to_csv(ATTENDANCE_FILE, index=False)

# --- FUNCTIONS ---
def load_known_faces():
    known_encodings = []
    known_names = []
    for filename in os.listdir(FACES_DIR):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            img = face_recognition.load_image_file(f"{FACES_DIR}/{filename}")
            encoding = face_recognition.face_encodings(img)[0]
            known_encodings.append(encoding)
            known_names.append(os.path.splitext(filename)[0])
    return known_encodings, known_names

def mark_attendance(name):
    df = pd.read_csv(ATTENDANCE_FILE)
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    # Check if already marked today
    if not ((df['Name'] == name) & (df['Date'] == date_str)).any():
        new_entry = pd.DataFrame([[name, date_str, time_str]], columns=["Name", "Date", "Time"])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)
        return True
    return False

# --- STREAMLIT UI ---
st.title("Smart Attendance System 📸")

menu = ["Take Attendance", "Register New Face", "View Records"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Take Attendance":
    st.header("Real-time Attendance")
    known_encodings, known_names = load_known_faces()
    
    # Use Streamlit's built-in camera input
    img_file = st.camera_input("Position your face in the frame")
    
    if img_file:
        bytes_data = img_file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        
        # Find faces
        face_locations = face_recognition.face_locations(rgb_img)
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
        
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]
                    if mark_attendance(name):
                        st.success(f"Attendance marked for: {name}")
                    else:
                        st.info(f"{name}, your attendance is already recorded for today.")
                else:
                    st.error("Face not recognized. Please register first.")

elif choice == "Register New Face":
    st.header("Register New User")
    new_name = st.text_input("Enter Name")
    reg_img = st.camera_input("Take a photo to register")
    
    if st.button("Save Profile") and new_name and reg_img:
        with open(f"{FACES_DIR}/{new_name}.jpg", "wb") as f:
            f.write(reg_img.getbuffer())
        st.success(f"Profile created for {new_name}!")

elif choice == "View Records":
    st.header("Attendance Logs")
    df = pd.read_csv(ATTENDANCE_FILE)
    st.dataframe(df, use_container_width=True)
    st.download_button("Download CSV", df.to_csv(index=False), "attendance.csv", "text/csv")
