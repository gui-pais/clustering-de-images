import os
import shutil

def delete_directory(directory_name):
    directory_path = os.path.abspath(directory_name)
    if os.path.exists(directory_path):
        try:
            shutil.rmtree(directory_path)
            print(f"Directory {directory_path} and its contents were successfully deleted.")
        except Exception as e:
            print(f"Error deleting directory {directory_path}: {e}")
    else:
        print(f"Directory {directory_path} does not exist.")
