# add_dates.py - Upload semester exam dates to Firestore

import csv
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore

# === Firebase Setup ===
FIREBASE_KEY_PATH = "firebase_key.json"
cred = credentials.Certificate(FIREBASE_KEY_PATH)
firebase_admin.initialize_app(cred)
db = firestore.client()

# === CSV File Path ===
CSV_PATH = "F:\\semester6_exam_schedule.csv"

# === Prepare and Upload Data ===
departments_data = {}

with open(CSV_PATH, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        dept = row["department"].strip()
        if dept not in departments_data:
            departments_data[dept] = []

        date_obj = datetime.strptime(row["date"], "%Y-%m-%d")

        departments_data[dept].append({
            "title": row["title"].strip(),
            "subject_code": row["subject_code"].strip(),
            "subject_name": row["subject_name"].strip(),
            "date": date_obj,
            "time": row["time"].strip(),
            "event_type": row["event_type"].strip(),
            "semester": row["semester"].strip(),
            "session": row["session"].strip(),
            "description": f"{row['subject_code']} - {row['title']} exam for {dept}"
        })

# Upload to Firestore
for dept, exams in departments_data.items():
    doc_ref = db.collection("important_dates").document(dept)
    doc_ref.set({"exam_dates": exams}, merge=True)

print("✅ Exam dates uploaded successfully!")
