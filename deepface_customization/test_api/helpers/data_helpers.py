import os

import pickle

from face_recognition.modules import detection

# DATA_PTH = os.path.join("./deepface_customization", "test_api", "data")
DATA_PTH = "./test_api/data"

def load_stored_data():
    stored_data_pth = os.path.join(DATA_PTH, "data.pkl")
    if os.path.isfile(stored_data_pth):
        with open(stored_data_pth, "rb") as file:
            stored_data = pickle.load(file)
            return stored_data
    
    data = {"X": [], "y": []}
    for dir in os.scandir(DATA_PTH):
        if dir.is_dir():
            label = dir.name
            dir_path = dir.path
            backup_file_pth = os.path.join(dir_path, "backup.pkl")
            
            if os.path.isfile(backup_file_pth):
                with open(backup_file_pth, "rb") as file:
                    X = pickle.load(file)
                    y = [label] * len(X)
                data["X"].extend(X)
                data["y"].extend(y)
            
            else:
                for file in os.listdir(dir_path):
                    if file.endswith(".jpg"):
                        img_pth = os.path.join(dir_path, file)
                        img_embeddings, _ = detection.extract_embeddings_and_facial_areas(
                            img_path = img_pth,
                            detector_backend = "retinaface"
                        )

                        y = [label] * len(img_embeddings)
                        data["X"].extend(img_embeddings)
                        data["y"].extend(y)

                    with open(os.path.join(dir_path, "backup.pkl"), "wb") as file:
                        pickle.dump(img_embeddings, file)
    
    with open(stored_data_pth, "wb") as file:
        pickle.dump(data, file)
    
    return data
