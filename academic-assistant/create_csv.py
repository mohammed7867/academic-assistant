import csv

# Data
events = [
    ['title', 'date', 'event_type', 'description'],
    ['Midterm Exam', '2025-04-15', 'exam', 'CS101 Midterm Examination'],
    ['Final Exam', '2025-05-30', 'exam', 'CS101 Final Examination'],
    ['Assignment 1', '2025-04-10', 'assignment', 'Submit via online portal'],
    ['Assignment 2', '2025-05-01', 'assignment', 'Group project submission'],
    ['Feedback Form', '2025-05-15', 'feedback', 'Course feedback submission']
]

# Writing to 'dates.csv'
with open('dates.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(events)

print("CSV file 'dates.csv' created successfully.")
