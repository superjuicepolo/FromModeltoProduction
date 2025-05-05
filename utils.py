import os

def get_jpg_image_paths(folder_path):
    jpg_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.jpg'):
                full_path = os.path.join(root, file)
                jpg_paths.append(full_path)
    return jpg_paths
