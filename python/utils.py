import os

def ensure_folder_exist(filename):
    """Ensure the folder for a file always exists. """
    os.makedirs(os.path.split(filename)[0], exist_ok=True)