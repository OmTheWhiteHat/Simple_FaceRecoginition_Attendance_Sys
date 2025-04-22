import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from threading import Thread
import sqlite3
import os
from hashlib import sha256
from datetime import datetime
import csv

class FaceRecognise:
    def __init__(self):
        self.is_running = False
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        if not os.path.exists("trainer.yml"):
            raise FileNotFoundError("trainer.yml not found. Please train the model first.")
        self.recognizer.read("trainer.yml")
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def string_to_int_id(self, s):
        return int(sha256(s.encode('utf-8')).hexdigest(), 16) % (10**8)

    def fetch_student_details(self, student_id):  # Corrected function definition
        try:
            conn = sqlite3.connect('students.db')
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM students")
            students = cursor.fetchall()

            for student in students:
                if self.string_to_int_id(student[0]) == student_id:
                    return student
        except Exception as e:
            print(f"[ERROR] Database error: {e}")
        finally:
            conn.close()
        return None

    def mark_attendance(self, student_id, name, roll_no):
        filename = "Attendance.csv"  # This should be indented under the function

        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        # Create CSV if it doesn't exist
        if not os.path.exists(filename):
            with open(filename, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Student ID", "Name", "Roll No", "Date", "Time"])

        # Check for duplicate entry for the same student on the same day
        with open(filename, "r", newline='') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header if exists
            for row in reader:
                # Check if the name, roll number, and date are the same
                if len(row) > 3 and row[1] == name and row[2] == roll_no and row[3] == date_str:
                    return  # If already marked today, return without marking again

        # Mark attendance only if not already marked
        with open(filename, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([student_id, name, roll_no, date_str, time_str])
            print(f"[INFO] Attendance marked for {name} on {date_str}")


    def recognize_faces(self):
        self.is_running = True
        cam = cv2.VideoCapture(0)

        if not cam.isOpened():
            messagebox.showerror("Error", "Unable to access the webcam.")
            return

        recognized_ids = set()  # Set to track students that have been processed

        while self.is_running:
            ret, frame = cam.read()
            if not ret:
                print("Failed to grab frame")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face_roi = gray[y:y + h, x:x + w]
                label, confidence = self.recognizer.predict(face_roi)

                if confidence < 80:
                    # Check if this student has been recognized already
                    if label not in recognized_ids:
                        recognized_ids.add(label)  # Add student ID to recognized set

                        # Fetch student details
                        student_details = self.fetch_student_details(label)
                        if student_details:
                            student_id, name, father_name, roll_no, address, contact_number, email, course, semester, branch, date_of_birth, gender = student_details

                            # Mark attendance only once per student
                            self.mark_attendance(student_id, name, roll_no)

                            # Draw rectangle around face and display student info
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (118, 8, 252), 2)
                            cv2.putText(frame, f"{name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                            cv2.putText(frame, f"{roll_no} | Sem: {semester}", (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (93, 0, 255), 2)
                            cv2.putText(frame, f"Branch: {branch}", (x, y + h + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (215, 215, 215), 2)
                     
                    else:
                            # Fetch student details
                            if student_details:
                                # Mark attendance only once per student
                                self.mark_attendance(student_id, name, roll_no)

                                # Draw rectangle around face and display student info
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (118, 8, 252), 2)
                                cv2.putText(frame, f"Already Marked", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (118, 8, 252), 2)
                                cv2.putText(frame, f"{name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                                cv2.putText(frame, f"{roll_no} | Sem: {semester}", (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (93, 0, 255), 2)
                                cv2.putText(frame, f"Branch: {branch}", (x, y + h + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (215, 215, 215), 2)
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Display the frame with detections
            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) == 27:  # ESC key to stop
                break

        cam.release()
        cv2.destroyAllWindows()



    def stop_recognition(self):
        self.is_running = False

    def exit_program(self):
        self.stop_recognition()
        self.window.quit()

    def start_gui(self):
        self.window = tk.Tk()
        self.window.title("Real-Time Face Recognition")
        self.window.geometry("400x250")

        status_label = tk.Label(self.window, text="Face Recognition System", font=("Arial", 16, "bold"))
        status_label.pack(pady=20)

        start_button = tk.Button(self.window, text="Start Recognition", width=20,
                                 command=lambda: Thread(target=self.recognize_faces, daemon=True).start())
        start_button.pack(pady=10)

        stop_button = tk.Button(self.window, text="Stop Recognition", width=20, command=self.stop_recognition)
        stop_button.pack(pady=10)

        exit_button = tk.Button(self.window, text="Exit", width=20, command=self.exit_program)
        exit_button.pack(pady=10)

        self.window.mainloop()

# Run the GUI
if __name__ == "__main__":
    app = FaceRecognise()
    app.start_gui()
