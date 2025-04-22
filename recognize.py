import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from threading import Thread
import sqlite3

# Load the trained recognizer and the labels
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Global variable to control webcam capture thread
is_running = False

# Function to fetch student details by student ID
def fetch_student_details(student_id):
    try:
        conn = sqlite3.connect('students.db')  # Connect to database
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM students WHERE student_id=?", (student_id,))
        student = cursor.fetchone()
        return student
    finally:
        conn.close()

# Function to start recognition
def recognize_faces():
    global is_running
    is_running = True
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        messagebox.showerror("Error", "Unable to access the webcam.")
        return

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

            if confidence < 80:  # Adjusted threshold for better accuracy
                student_details = fetch_student_details(label)
                if student_details:
                    student_id, student_name, father_name, roll_no, address, course, semester, branch = student_details

                    # Display name and course info on screen
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{student_name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    cv2.putText(frame, f"{course}, Sem: {semester}", (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)
                else:
                    cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) == 27:  # ESC key to stop
            break

    cam.release()
    cv2.destroyAllWindows()

# Stop recognition safely
def stop_recognition():
    global is_running
    is_running = False

# Exit app and stop recognition if running
def exit_program():
    stop_recognition()
    window.quit()

# Create GUI using Tkinter
window = tk.Tk()
window.title("Real-Time Face Recognition")
window.geometry("400x250")

status_label = tk.Label(window, text="Face Recognition System", font=("Arial", 16, "bold"))
status_label.pack(pady=20)

start_button = tk.Button(window, text="Start Recognition", width=20, command=lambda: Thread(target=recognize_faces, daemon=True).start())
start_button.pack(pady=10)

stop_button = tk.Button(window, text="Stop Recognition", width=20, command=stop_recognition)
stop_button.pack(pady=10)

exit_button = tk.Button(window, text="Exit", width=20, command=exit_program)
exit_button.pack(pady=10)

window.mainloop()
