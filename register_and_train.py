import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import ttk
import sqlite3
from tkinter import PhotoImage
import re
from hashlib import sha256


def exit_program():
    window.quit()

def create_student_table():
    # Connect to the database (or create it if it doesn't exist)
    conn = sqlite3.connect('students.db')
    c = conn.cursor()

    # Create the students table
    c.execute('''
    CREATE TABLE IF NOT EXISTS students (
        student_id VARCHAR(10) PRIMARY KEY,
        name TEXT NOT NULL,
        father_name TEXT,
        roll_no VARCHAR(10) UNIQUE,
        address TEXT,
        contact_number VARCHAR(15),
        email TEXT UNIQUE,
        course TEXT,
        semester TEXT,
        branch TEXT,
        date_of_birth TEXT,
        gender TEXT
    )
''')


    conn.commit()
    conn.close()

# Call the function to create the table when running the script for the first time
create_student_table()

def insert_student_details(student_id, name, father_name, roll_no, address, contact_number, email, course, semester, branch, date_of_birth, gender):
    try:
        conn = sqlite3.connect('students.db')
        c = conn.cursor()
        c.execute('''
            INSERT INTO students (student_id, name, father_name, roll_no, address, contact_number, email, course, semester, branch, date_of_birth, gender)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (student_id, name, father_name, roll_no, address, contact_number, email, course, semester, branch, date_of_birth, gender))
        conn.commit()
        conn.close()
    except sqlite3.IntegrityError as e:
        messagebox.showerror("Database Error", f"Error inserting data: {e}")

#update
def update_student_details(student_id, name, father_name, roll_no, address, contact_number, email, course, semester, branch, date_of_birth, gender):
    try:
        conn = sqlite3.connect('students.db')
        c = conn.cursor()

        # Ensure the student exists before updating
        c.execute("SELECT * FROM students WHERE student_id = ?", (student_id,))
        if c.fetchone() is None:
            messagebox.showwarning("Not Found", f"No student found with ID {student_id}")
            return

        # Update statement
        c.execute('''
            UPDATE students
            SET name = ?, father_name = ?, roll_no = ?, address = ?, contact_number = ?, email = ?,
                course = ?, semester = ?, branch = ?, date_of_birth = ?, gender = ?
            WHERE student_id = ?
        ''', (name, father_name, roll_no, address, contact_number, email,
              course, semester, branch, date_of_birth, gender, student_id))

        conn.commit()
        conn.close()
        messagebox.showinfo("Success", f"Student ID {student_id} updated successfully.")

    except sqlite3.Error as e:
        messagebox.showerror("Database Error", f"Error updating data: {e}")

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

# # Function to capture and train face images for a student
# def fetch_student_details(student_id):
#     """Fetch student details from the database based on student ID."""
#     try:
#         conn = sqlite3.connect("students.db")
#         cursor = conn.cursor()
#         cursor.execute("SELECT * FROM students WHERE student_id=?", (student_id,))
#         student = cursor.fetchone()
#         conn.close()

#         if student:
#             return {
#                 "student_id": student[0],
#                 "name": student[1],
#                 "father_name": student[2],
#                 "roll_no": student[3],
#                 "address": student[4],
#                 "course": student[5],
#                 "semester": student[6],
#                 "branch": student[7]
#             }
#         else:
#             return None
#     except Exception as e:
#         print(f"Error fetching student details: {e}")
#         return None
# train model

def string_to_int_id(s):
    """Converts a string ID into a unique integer using hashing."""
    return int(sha256(s.encode('utf-8')).hexdigest(), 16) % (10**8)

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

        # Skip if not a directory
        if not os.path.isdir(folder_path):
            continue

        # Skip folders with spaces or invalid characters
        if " " in folder_name or not re.match(r"^[\w-]+$", folder_name):
            print(f"[WARNING] Skipping invalid folder: {folder_name}")
            continue

        student_id = folder_name
        int_id = string_to_int_id(student_id)

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
                ids.append(int_id)

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
#save 
def on_save_button_click():
    student_id = student_id_entry.get()
    name = name_entry.get()
    father_name = father_name_entry.get()
    roll_no = roll_no_entry.get()
    address = address_entry.get()
    contact_number = contact_number_entry.get()
    email = email_entry.get()
    date_of_birth = dob_entry.get()
    gender = gender_combobox.get()
    course = course_combobox.get()
    semester = semester_combobox.get()
    branch = branch_combobox.get()

    if not all([student_id, name, father_name, roll_no, address, contact_number, email, date_of_birth, gender, course, semester, branch]):
        messagebox.showwarning("Input Error", "Please fill in all the details.")
        return

    try:
        # Check if student exists
        conn = sqlite3.connect('students.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM students WHERE student_id = ?", (student_id,))
        existing_student = cursor.fetchone()
        conn.close()

        if existing_student:
            update_student_details(student_id, name, father_name, roll_no, address, contact_number, email, course, semester, branch, date_of_birth, gender)
        else:
            insert_student_details(student_id, name, father_name, roll_no, address, contact_number, email, course, semester, branch, date_of_birth, gender)

        update_student_table()

    except Exception as e:
        messagebox.showerror("Error", f"Failed to save or update student: {e}")
        
def reset_form_fields():
    # Clear all entry fields
    student_id_entry.delete(0, tk.END)
    name_entry.delete(0, tk.END)
    father_name_entry.delete(0, tk.END)
    roll_no_entry.delete(0, tk.END)
    gender_combobox.set('')
    dob_entry.delete(0, tk.END)
    address_entry.delete(0, tk.END)
    contact_number_entry.delete(0, tk.END)
    email_entry.delete(0, tk.END)

    # Reset top filters
    course_combobox.set('')
    semester_combobox.set('')
    branch_combobox.set('')

    # Optionally set focus back to Student ID
    student_id_entry.focus()


def on_table_row_click(event):
    selected_item = student_table.selection()
    if not selected_item:
        return
    data = student_table.item(selected_item)['values']
    
    # Fill the form fields
    student_id_entry.delete(0, tk.END)
    student_id_entry.insert(0, data[0])
    name_entry.delete(0, tk.END)
    name_entry.insert(0, data[1])
    father_name_entry.delete(0, tk.END)
    father_name_entry.insert(0, data[2])
    roll_no_entry.delete(0, tk.END)
    roll_no_entry.insert(0, data[3])
    address_entry.delete(0, tk.END)
    address_entry.insert(0, data[4])
    contact_number_entry.delete(0, tk.END)
    contact_number_entry.insert(0, data[5])
    email_entry.delete(0, tk.END)
    email_entry.insert(0, data[6])
    course_combobox.set(data[7])
    semester_combobox.set(data[8])
    branch_combobox.set(data[9])
    dob_entry.delete(0, tk.END)
    dob_entry.insert(0, data[10])
    gender_combobox.set(data[11])

    # Add similar lines for other fields if shown in the table

#execute capture and train
def execute_capture_and_train():
    student_id = student_id_entry.get()
    capture_images(student_id)
    train_model()

# Tkinter Window
# Colors and styles
PRIMARY_COLOR = "#3f00a3"
SECONDARY_COLOR = "#F4F4F9"
ACCENT_COLOR = "#4CAF50"
TEXT_COLOR = "white"
FONT = ("Segoe UI", 11)

# Main Window
window = tk.Tk()
window.title("Student Registration System")
window.geometry("1200x700")
window.configure(bg=SECONDARY_COLOR)

# PanedWindow
paned_window = tk.PanedWindow(window, orient="horizontal", bg=SECONDARY_COLOR)
paned_window.pack(fill=tk.BOTH, expand=True)

# Top frame for filters
top_frame = tk.LabelFrame(window, text="Student Filter", font=("Segoe UI", 12, "bold"),
                          bg=PRIMARY_COLOR, fg=TEXT_COLOR, bd=2, relief=tk.RIDGE)
top_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=10)

def add_combobox(frame, text, row, values):
    label = tk.Label(frame, text=text, font=FONT, bg=PRIMARY_COLOR, fg=TEXT_COLOR)
    label.grid(row=row, column=0, sticky="w", pady=5, padx=10)
    combobox = ttk.Combobox(frame, font=FONT, values=values, state="readonly")
    combobox.grid(row=row, column=1, pady=5, padx=10)
    return combobox

course_combobox = add_combobox(top_frame, "Course", 0, ["DIPLOMA"])
semester_combobox = add_combobox(top_frame, "Semester", 1, [f"Semester {i}" for i in range(1, 7)])
branch_combobox = add_combobox(top_frame, "Branch", 2, ["Computer Science", "Mechanical", "Civil", "Electrical", "Electronics"])

# Left Frame for form
left_frame = tk.LabelFrame(paned_window, text="Student Registration", font=("Segoe UI", 12, "bold"),
                           bg=SECONDARY_COLOR, fg="black", bd=2, relief=tk.RIDGE)
paned_window.add(left_frame, minsize=500)

def add_entry(row, label_text, parent=left_frame, values=None):
    label = tk.Label(parent, text=label_text, font=FONT, bg=SECONDARY_COLOR, fg='black')
    label.grid(row=row, column=0, sticky="w", pady=8, padx=10)
    if values:
        entry = ttk.Combobox(parent, font=FONT, values=values, state="readonly")
    else:
        entry = tk.Entry(parent, font=FONT)
    entry.grid(row=row, column=1, pady=8, padx=10, sticky="ew")
    return entry

student_id_entry = add_entry(0, "Student ID")
name_entry = add_entry(1, "Name")
father_name_entry = add_entry(2, "Father's Name")
roll_no_entry = add_entry(3, "Roll Number")
gender_combobox = add_entry(4, "Gender", values=["Male", "Female", "Other"])
dob_entry = add_entry(5, "Date of Birth")
address_entry = add_entry(6, "Address")
contact_number_entry = add_entry(7, "Contact Number")
email_entry = add_entry(8, "Email")

# Register and Exit buttons
button_frame = tk.Frame(left_frame, bg=SECONDARY_COLOR)
button_frame.grid(row=9, columnspan=2, pady=10, padx=20)

save_button = tk.Button(button_frame, text="Save", font=FONT, bg=ACCENT_COLOR, fg=TEXT_COLOR, width=12, command=on_save_button_click)
save_button.pack(side=tk.LEFT, padx=5)

capture_button = tk.Button(button_frame, text="Capture", font=FONT, bg=ACCENT_COLOR, fg=TEXT_COLOR, width=12, command=execute_capture_and_train)
capture_button.pack(side=tk.LEFT, padx=5)

exit_button = tk.Button(button_frame, text="Exit", font=FONT, bg=ACCENT_COLOR, fg=TEXT_COLOR, width=12, command=window.destroy)
exit_button.pack(side=tk.LEFT, padx=5)

reset_button = tk.Button(button_frame, text="Reset", font=FONT, bg=ACCENT_COLOR, fg=TEXT_COLOR, width=12, command=reset_form_fields)
reset_button.pack(side=tk.LEFT, padx=5)


# Right Frame for student table
right_frame = tk.LabelFrame(paned_window, text="Registered Students", font=("Segoe UI", 12, "bold"),
                            bg=SECONDARY_COLOR, fg="black", bd=2, relief=tk.RIDGE)
paned_window.add(right_frame)

columns = ("Student ID", "Name", "Father's Name", "Roll No", "Address", "Course", "Semester", "Branch")
student_table = ttk.Treeview(right_frame, columns=columns, show="headings")
student_table.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
student_table.tag_configure('oddrow', background="#f2f2f2")
student_table.tag_configure('evenrow', background="#ffffff")
student_table.bind("<ButtonRelease-1>", on_table_row_click)

student_data = fetch_all_student_details()
for index, data in enumerate(student_data):
    tag = 'evenrow' if index % 2 == 0 else 'oddrow'
    student_table.insert("", "end", values=data, tags=(tag,))


for col in columns:
    student_table.heading(col, text=col)
    student_table.column(col, width=120, anchor="center")

# Optional: Add style to make Treeview look more modern
# Treeview style customization
style = ttk.Style()
style.theme_use("default")  # You can try "clam", "alt", or "default"

# Treeview heading style
style.configure("Treeview.Heading",
                font=("Segoe UI", 10, "bold"),
                background="#3f00a3",  # heading background
                foreground="white")    # heading text

# Treeview row style
style.configure("Treeview",
                background="white",
                foreground="black",
                rowheight=30,
                fieldbackground="white")

# Selected row style
style.map("Treeview",
          background=[('selected', '#b3d9ff')],
          foreground=[('selected', 'black')])



# 🔧 Add this to show table data at startup
update_student_table()
# Status Label for capturing and training
status_label = tk.Label(window, text="Status: Waiting for input...", font=("Arial", 12), bg='#F4F4F9', fg='black')
status_label.pack(pady=10)

# Add Scrollbar to the right of the table
scrollbar = ttk.Scrollbar(right_frame, orient="horizontal", command=student_table.xview)
student_table.configure(xscroll=scrollbar.set)
scrollbar.pack(side="bottom", fill="x")


# Run the Tkinter event loop
window.mainloop()
