import pandas as pd
from datetime import datetime
import os
from loges import logger

class DatabaseManager:
    def __init__(self, storage_path=None):
        self.storage_path = storage_path or os.getcwd()
        self.attendance_file = os.path.join(self.storage_path, "attendance.csv")
        logger.info(f"Database manager initialized with storage path: {self.storage_path}")
        self.initialize_attendance_file()

    def initialize_attendance_file(self):
        # Ensure the directory exists
        os.makedirs(self.storage_path, exist_ok=True)
        logger.info(f"Ensured storage directory exists: {self.storage_path}")
        
        if not os.path.exists(self.attendance_file):
            df = pd.DataFrame(columns=['Name', 'Date', 'Time'])
            df.to_csv(self.attendance_file, index=False)
            logger.info(f"Created new attendance file: {self.attendance_file}")
        else:
            logger.info(f"Using existing attendance file: {self.attendance_file}")

    def mark_attendance(self, name):
        try:
            df = pd.read_csv(self.attendance_file)
            now = datetime.now()
            date = now.strftime('%Y-%m-%d')
            time = now.strftime('%H:%M:%S')
            
            today_attendance = df[(df['Name'] == name) & (df['Date'] == date)]
            if len(today_attendance) == 0:
                new_row = pd.DataFrame([{
                    'Name': name,
                    'Date': date,
                    'Time': time
                }])
                df = pd.concat([df, new_row], ignore_index=True)
                df.to_csv(self.attendance_file, index=False)
                logger.info(f"Attendance marked for {name} at {date} {time}")
                return True
            else:
                logger.info(f"Attendance already marked for {name} today")
                return False
        except Exception as e:
            logger.error(f"Error marking attendance: {str(e)}")
            return False
