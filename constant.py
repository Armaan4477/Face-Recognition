import json
import platform
import os
from PyQt6.QtCore import QThread, pyqtSignal
from loges import logger

class ConfigManager(QThread):
    config_updated = pyqtSignal(dict)
    config_ready = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.config_file_name = ".config.json"
        self.current_version = "1.0.0"
        self.config_file = self.get_config_file_path()

    def get_config_file_path(self):
        if platform.system() == 'Windows':
            cache_dir = os.path.join(os.getenv('APPDATA'), 'Face-Recog')
        elif platform.system() == 'Linux':
            cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'Face-Recog')
        elif platform.system() == 'Darwin':
            cache_dir = os.path.join(os.path.expanduser('~/Library/Application Support'), 'Face-Recog')
        else:
            logger.error("Unsupported OS!")
            return None

        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Config directory created/ensured: {cache_dir}")
        return os.path.join(cache_dir, self.config_file_name)

    def get_default_path(self):
        if platform.system() == 'Windows':
            file_path = "C:\\Attendance Data"
        elif platform.system() == 'Linux':
            home_dir = os.path.expanduser('~')
            os.makedirs(os.path.join(home_dir, "Attendance Data"), exist_ok=True)
            file_path = os.path.join(home_dir, "Attendance Data")
        elif platform.system() == 'Darwin':
            home_dir = os.path.expanduser('~')
            documents_dir = os.path.join(home_dir, "Documents")
            os.makedirs(os.path.join(documents_dir, "Attendance Data"), exist_ok=True)
            file_path = os.path.join(documents_dir, "Attendance Data")
        else:
            logger.error("Unsupported OS!")
            file_path = None
        logger.info(f"Default path determined: {file_path}")
        return file_path

    def write_config(self, data):
        with open(self.config_file, 'w') as file:
            json.dump(data, file, indent=4)
        logger.info(f"Configuration written to {self.config_file}")
        self.config_updated.emit(data)

    def get_config(self):
        try:
            with open(self.config_file, 'r') as file:
                data = json.load(file)
            logger.info(f"Loaded configuration from {self.config_file}")
            return data
        except FileNotFoundError:
            logger.warning(f"Configuration file {self.config_file} not found. Returning empty config.")
            return {}

    def run(self):
        if not os.path.exists(self.config_file):
            file_path = self.get_default_path()
            default_config = {
                "version": self.current_version,
                "save_to_directory": file_path
            }
            self.write_config(default_config)
            logger.info("Created new configuration file.")
        else:
            config_data = self.get_config()
            if "version" not in config_data or config_data["version"] != self.current_version:
                logger.info("Configuration version mismatch or missing. Overwriting with default config.")
                save_to_directory = config_data.get("save_to_directory", self.get_default_path())

                default_config = {
                    "version": self.current_version,
                    "save_to_directory": save_to_directory
                }
                self.write_config(default_config)
            else:
                logger.info(f"Loaded configuration: {config_data}")
                self.config_updated.emit(config_data)
        self.config_ready.emit()
        self.config_ready.emit()
