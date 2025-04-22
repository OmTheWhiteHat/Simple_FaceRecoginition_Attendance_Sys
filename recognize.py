import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from threading import Thread
import sqlite3
import os
from hashlib import sha256

# --- Hashing function (used in training) ---
def string_to_int_id(s):
    return int(sha256(s.encode('utf-8')).hexdigest(), 16) % (10**8)

# Load the recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
if not os.path.exists("trainer.yml"):
    raise FileNotFoundError("trainer.yml not found. Please train the model first.")
recognizer.read("trainer.yml")

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Control webcam thread
is_running = False

# Fetch student details from SQLite DB
def fetch_student_details(student_id):
    try:
        conn = sqlite3.connect('students.db')
        cursor = conn.cursor()

        # Accept both string and int form
        cursor.execute("SELECT * FROM students")
        students = cursor.fetchall()

        for student in students:
            if string_to_int_id(student[0]) == student_id:
                return student
    except Exception as e:
        print(f"[ERROR] Database error: {e}")
    finally:
        conn.close()
    return None

# Recognize face and match with DB
def recognize_faces():
    global is_running
    is_running = True
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        messagebox.showerror("Error", "Unable to access the webcam.")
        return

    recognized_ids = set()  # To store already recognized student IDs during the session

    while is_running:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            label, confidence = recognizer.predict(face_roi)

            if confidence < 80:
                if label not in recognized_ids:
                    recognized_ids.add(label)  # Add this label to the recognized set

                student_details = fetch_student_details(label)
                if student_details:
                    student_id, name, father_name, roll_no, address, contact_number, email, course, semester, branch, date_of_birth, gender = student_details

                    # Draw green rectangle and info
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (118, 8, 252), 2)
                    cv2.putText(frame, f"{name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    cv2.putText(frame, f"{roll_no} | Sem: {semester}", (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (93, 0, 255), 2)
                    cv2.putText(frame, f"Branch: {branch}", (x, y + h + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (215, 215, 215), 2)
                else:
                    # If no match found
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 149, 237), 2)
                    cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                # Confidence too low
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) == 27:  # ESC key
            break

    cam.release()
    cv2.destroyAllWindows()

# Stop recognition safely
def stop_recognition():
    global is_running
    is_running = False

# Exit the app
def exit_program():
    stop_recognition()
    window.quit()

# Tkinter GUI
window = tk.Tk()
window.title("Real-Time Face Recognition")
window.geometry("400x250")

status_label = tk.Label(window, text="Face Recognition System", font=("Arial", 16, "bold"))
status_label.pack(pady=20)

start_button = tk.Button(window, text="Start Recognition", width=20,
                         command=lambda: Thread(target=recognize_faces, daemon=True).start())
start_button.pack(pady=10)

stop_button = tk.Button(window, text="Stop Recognition", width=20, command=stop_recognition)
stop_button.pack(pady=10)

exit_button = tk.Button(window, text="Exit", width=20, command=exit_program)
exit_button.pack(pady=10)

window.mainloop()
