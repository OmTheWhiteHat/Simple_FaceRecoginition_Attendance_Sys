import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import sqlite3
from tkinter import PhotoImage

def exit_program():
    window.quit()

def create_student_table():
    # Connect to the database (or create it if it doesn't exist)
    conn = sqlite3.connect('students.db')
    c = conn.cursor()

    # Create the students table
    c.execute('''
        CREATE TABLE IF NOT EXISTS students (
            student_id varchar(10) PRIMARY KEY,
            name TEXT,
            father_name TEXT,
            roll_no varchar(10) unique,
            address TEXT,
            course TEXT,
            semester TEXT,
            branch TEXT
        )
    ''')

    conn.commit()
    conn.close()

# Call the function to create the table when running the script for the first time
create_student_table()

def insert_student_details(student_id, name, father_name, roll_no, address, course, semester, branch):
    try:
        conn = sqlite3.connect('students.db')
        c = conn.cursor()
        c.execute('''
            INSERT INTO students (student_id, name, father_name, roll_no, address, course, semester, branch)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (student_id, name, father_name, roll_no, address, course, semester, branch))
        conn.commit()
        conn.close()
    except sqlite3.IntegrityError as e:
        messagebox.showerror("Database Error", f"Error inserting data: {e}")



# Old function (conflict)
# def fetch_student_details():

# ✅ New name to avoid conflict
def fetch_all_student_details():
    conn = sqlite3.connect('students.db')
    c = conn.cursor()
    c.execute("SELECT * FROM students")
    students = c.fetchall()
    conn.close()
    return students



# Function to update the table with student data
def update_student_table():
    for row in student_table.get_children():
        student_table.delete(row)

    students = fetch_all_student_details()  # ✅ Fixed name
    for student in students:
        student_table.insert("", "end", values=student)

# Function to capture and train face images for a student
def fetch_student_details(student_id):
    """Fetch student details from the database based on student ID."""
    try:
        conn = sqlite3.connect("students.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM students WHERE student_id=?", (student_id,))
        student = cursor.fetchone()
        conn.close()

        if student:
            return {
                "student_id": student[0],
                "name": student[1],
                "father_name": student[2],
                "roll_no": student[3],
                "address": student[4],
                "course": student[5],
                "semester": student[6],
                "branch": student[7]
            }
        else:
            return None
    except Exception as e:
        print(f"Error fetching student details: {e}")
        return None
# train model
def train_model():
    print("[INFO] Starting training...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    face_samples = []
    ids = []

    base_path = "train_images"
    if not os.path.exists(base_path):
        print("[ERROR] train_images folder not found.")
        return

    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)

        if not os.path.isdir(folder_path):
            continue

        try:
            student_id = int(folder_name)
        except ValueError:
            print(f"[WARNING] Skipping non-numeric folder: {folder_name}")
            continue

        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[WARNING] Could not read image: {img_path}")
                continue

            face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:
                print(f"[WARNING] No faces found in image: {img_path}")
                continue

            for (x, y, w, h) in faces:
                face_samples.append(img[y:y+h, x:x+w])
                ids.append(student_id)

    if len(face_samples) == 0:
        print("[ERROR] No valid training images found.")
        return
    else:
        print(f"[INFO] Found {len(face_samples)} face samples. Training now...")

    recognizer.train(face_samples, np.array(ids))
    recognizer.save("trainer.yml")
    print("[INFO] Trainer model saved to trainer.yml.")


def capture_images(student_id):
    cam = cv2.VideoCapture(0)
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    count = 0
    save_path = f"train_images/{student_id}"
    os.makedirs(save_path, exist_ok=True)
    if not os.path.exists(save_path):
        print(f"Folder for student {student_id} does not exist, creating it now.")
        os.makedirs(save_path, exist_ok=True)

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to capture image")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            print("No faces detected in the captured image.")


        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))

            # Save the image to the student's folder
            cv2.imwrite(f"{save_path}/User.{student_id}.{count}.jpg", face_img)
            print(f"Captured image {count} for student {student_id}")


            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"Image {count}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow('Capturing Faces - Press ESC to Exit', frame)

        if cv2.waitKey(100) & 0xff == 27 or count >= 30:
            break

    cam.release()
    cv2.destroyAllWindows()


# Define the function that is called when the register button is clicked
def on_register_button_click():
    student_id = student_id_entry.get()
    name = name_entry.get()
    father_name = father_name_entry.get()
    roll_no = roll_no_entry.get()
    address = address_entry.get()
    course = course_combobox.get()
    semester = semester_combobox.get()
    branch = branch_combobox.get()

    if student_id == "" or name == "" or father_name == "" or roll_no == "" or address == "" or course == "" or semester == "" or branch == "":
        messagebox.showwarning("Input Error", "Please fill in all the details.")
    else:
        # Insert student details into the database
        insert_student_details(student_id, name, father_name, roll_no, address, course, semester, branch)
        
        # Capture and train face
        capture_images(student_id)
        train_model()
        # Update the student table with new data
        update_student_table()

# Setup Tkinter window
window = tk.Tk()
window.title("Student Registration System")
window.geometry("1000x600")
window.configure(bg='#F4F4F9')  # Set background color

# Create a PanedWindow to split the UI into two sections
paned_window = tk.PanedWindow(window, orient="horizontal", bg='#F4F4F9')
paned_window.pack(fill=tk.BOTH, expand=True)

# Create left panel for form inputs
left_frame = tk.Frame(paned_window, bg='#F4F4F9')
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=20)

# Add UI elements for student input with modern styling
student_id_label = tk.Label(left_frame, text="Student ID", font=("Arial", 12), bg='#F4F4F9', fg='black')
student_id_label.grid(row=0, column=0, sticky="w", pady=5)
student_id_entry = tk.Entry(left_frame, font=("Arial", 12))
student_id_entry.grid(row=0, column=1, pady=5)

name_label = tk.Label(left_frame, text="Name", font=("Arial", 12), bg='#F4F4F9', fg='black')
name_label.grid(row=1, column=0, sticky="w", pady=5)
name_entry = tk.Entry(left_frame, font=("Arial", 12))
name_entry.grid(row=1, column=1, pady=5)

father_name_label = tk.Label(left_frame, text="Father's Name", font=("Arial", 12), bg='#F4F4F9', fg='black')
father_name_label.grid(row=2, column=0, sticky="w", pady=5)
father_name_entry = tk.Entry(left_frame, font=("Arial", 12))
father_name_entry.grid(row=2, column=1, pady=5)

roll_no_label = tk.Label(left_frame, text="Roll Number", font=("Arial", 12), bg='#F4F4F9', fg='black')
roll_no_label.grid(row=3, column=0, sticky="w", pady=5)
roll_no_entry = tk.Entry(left_frame, font=("Arial", 12))
roll_no_entry.grid(row=3, column=1, pady=5)

address_label = tk.Label(left_frame, text="Address", font=("Arial", 12), bg='#F4F4F9', fg='black')
address_label.grid(row=4, column=0, sticky="w", pady=5)
address_entry = tk.Entry(left_frame, font=("Arial", 12))
address_entry.grid(row=4, column=1, pady=5)

# Drop-down for Course
course_label = tk.Label(left_frame, text="Course", font=("Arial", 12), bg='#F4F4F9', fg='black')
course_label.grid(row=5, column=0, sticky="w", pady=5)
course_combobox = ttk.Combobox(left_frame, font=("Arial", 12), values=["B.Tech", "M.Tech", "BCA", "MCA", "BBA", "MBA"])
course_combobox.grid(row=5, column=1, pady=5)

# Drop-down for Semester
semester_label = tk.Label(left_frame, text="Semester", font=("Arial", 12), bg='#F4F4F9', fg='black')
semester_label.grid(row=6, column=0, sticky="w", pady=5)
semester_combobox = ttk.Combobox(left_frame, font=("Arial", 12), values=["Semester 1", "Semester 2", "Semester 3", "Semester 4", "Semester 5", "Semester 6", "Semester 7", "Semester 8"])
semester_combobox.grid(row=6, column=1, pady=5)

# Drop-down for Branch
branch_label = tk.Label(left_frame, text="Branch", font=("Arial", 12), bg='#F4F4F9', fg='black')
branch_label.grid(row=7, column=0, sticky="w", pady=5)
branch_combobox = ttk.Combobox(left_frame, font=("Arial", 12), values=["Computer Science", "Mechanical", "Civil", "Electrical", "Electronics", "Biotech"])
branch_combobox.grid(row=7, column=1, pady=5)

# Register Button
register_button = tk.Button(left_frame, text="Register", font=("Arial", 12), bg="#4CAF50", fg="white", command=on_register_button_click)
register_button.grid(row=8, column=0, columnspan=2, pady=10)

exit_button = tk.Button(left_frame, text="Exit", font=("Arial", 12), bg="#4CAF50", fg="white", command=exit_program)
exit_button.grid(row=8, column=1, columnspan=1, pady=10)

# Create a PanedWindow for the student table on the right
right_frame = tk.Frame(paned_window, bg='#F4F4F9')
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=20, pady=20)

# Create student details table
student_table = ttk.Treeview(right_frame, columns=("Student ID", "Name", "Father's Name", "Roll No", "Address", "Course", "Semester", "Branch"), show="headings")
student_table.pack(fill=tk.BOTH, expand=True)

# Define headings for the student table
student_table.heading("Student ID", text="Student ID")
student_table.heading("Name", text="Name")
student_table.heading("Father's Name", text="Father's Name")
student_table.heading("Roll No", text="Roll No")
student_table.heading("Address", text="Address")
student_table.heading("Course", text="Course")
student_table.heading("Semester", text="Semester")
student_table.heading("Branch", text="Branch")

# 🔧 Add this to show table data at startup
update_student_table()

# Status Label for capturing and training
status_label = tk.Label(window, text="Status: Waiting for input...", font=("Arial", 12), bg='#F4F4F9', fg='black')
status_label.pack(pady=10)

# Add Scrollbar to the right of the table
scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=student_table.yview)
student_table.configure(yscroll=scrollbar.set)
scrollbar.pack(side="right", fill="y")


# Run the Tkinter event loop
window.mainloop()
