import pandas as pd
from datetime import datetime
import os

class DatabaseManager:
    def __init__(self):
        self.attendance_file = "attendance.csv"
        self.initialize_attendance_file()

    def initialize_attendance_file(self):
        if not os.path.exists(self.attendance_file):
            df = pd.DataFrame(columns=['Name', 'Date', 'Time'])
            df.to_csv(self.attendance_file, index=False)

    def mark_attendance(self, name):
        try:
            df = pd.read_csv(self.attendance_file)
            now = datetime.now()
            date = now.strftime('%Y-%m-%d')
            time = now.strftime('%H:%M:%S')
            
            # Check if attendance already marked for today
            today_attendance = df[(df['Name'] == name) & (df['Date'] == date)]
            if len(today_attendance) == 0:
                new_row = pd.DataFrame([{
                    'Name': name,
                    'Date': date,
                    'Time': time
                }])
                # Use concat instead of append
                df = pd.concat([df, new_row], ignore_index=True)
                df.to_csv(self.attendance_file, index=False)
                return True
            return False
        except Exception as e:
            print(f"Error marking attendance: {str(e)}")
            return False
