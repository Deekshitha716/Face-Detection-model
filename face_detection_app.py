import streamlit as st
import cv2
import numpy as np
import pickle
import os
from deepface import DeepFace
from ultralytics import YOLO
import time

os.makedirs("database", exist_ok=True)

EMBEDDING_FILE = "database/embeddings.pkl"
LOG_FILE = "database/logs.txt"

# Load existing students
def load_database():
    if os.path.exists(EMBEDDING_FILE):
        with open(EMBEDDING_FILE, "rb") as f:
            return pickle.load(f)
    return {}

st.title("AI Exam Proctoring System")

menu = st.sidebar.selectbox(
    "Menu",
    ["Register Student", "Start Monitoring"]
)

# ---------------------------------------------------
# REGISTER STUDENT
# ---------------------------------------------------

if menu == "Register Student":

    st.header("Register New Student")

    student_name = st.text_input("Enter Student Name")

    uploaded_files = st.file_uploader(
        "Upload 3-5 clear face images",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True
    )

    if st.button("Register"):
        if student_name and uploaded_files:

            embeddings = []

            for file in uploaded_files:
                file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, 1)

                emb = DeepFace.represent(
                    img_path=img,
                    model_name="Facenet",
                    detector_backend="opencv",
                    enforce_detection=False
                )[0]["embedding"]

                embeddings.append(emb)

            student_db = load_database()
            student_db[student_name] = embeddings

            with open(EMBEDDING_FILE, "wb") as f:
                pickle.dump(student_db, f)

            st.success(f"{student_name} Registered Successfully!")

            st.experimental_rerun()

            st.success(f"{student_name} Registered Successfully!")

        else:
            st.warning("Enter name and upload images.")


# ---------------------------------------------------
# START MONITORING
# ---------------------------------------------------

elif menu == "Start Monitoring":

    st.header("Exam Monitoring")

    student_db = load_database()
    if len(student_db) == 0:
        st.warning("No students registered.")
        st.stop()

    selected_student = st.selectbox(
        "Select Registered Student",
        list(student_db.keys())
    )

    start_button = st.button("Start Exam")

    if start_button:

        model = YOLO("yolov8n.pt")
        registered_embeddings = student_db[selected_student]

        cap = cv2.VideoCapture(0)
        frame_window = st.empty()

        THRESHOLD = 0.8
        last_log_time = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (416, 320))

            # ---------------- PERSON DETECTION ----------------
            results = model(frame, verbose=False)
            person_count = 0

            for result in results:
                for box in result.boxes:
                    if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.5:
                        person_count += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            if person_count == 0:
                status_text = "No Candidate"
                status_color = (0, 0, 255)

            elif person_count > 1:
                status_text = "Multiple Persons"
                status_color = (0, 0, 255)

            else:
                status_text = "Monitoring..."   #i.e., person_count == 1
                status_color = (0, 255, 0)

            cv2.putText(frame, status_text,
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, status_color, 2)

            # ---------------- FACE VERIFICATION ----------------
            faces = DeepFace.extract_faces(
                img_path=frame,
                detector_backend="opencv",
                enforce_detection=False
            )

            for face in faces:

                area = face["facial_area"]
                x, y, w, h = area["x"], area["y"], area["w"], area["h"]

                face_img = (face["face"] * 255).astype(np.uint8)

                embedding = DeepFace.represent(
                    img_path=face_img,
                    model_name="Facenet",
                    detector_backend="opencv",
                    enforce_detection=False
                )[0]["embedding"]

                emb = np.array(embedding)
                emb = emb / np.linalg.norm(emb)

                distances = []

                for reg_emb in registered_embeddings:
                    reg = np.array(reg_emb)
                    reg = reg / np.linalg.norm(reg)
                    distances.append(np.linalg.norm(reg - emb))

                min_distance = min(distances)

                if min_distance < THRESHOLD:
                    label = selected_student
                    color = (0, 255, 0)
                else:
                    label = "Unknown"
                    color = (0, 0, 255)

                    # Log suspicious event
                    if time.time() - last_log_time > 5:
                        with open(LOG_FILE, "a") as log:
                            log.write(
                                f"{time.ctime()} - Unknown Person Detected\n"
                            )
                        last_log_time = time.time()

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label,
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)

            frame_window.image(frame, channels="BGR")

        cap.release()