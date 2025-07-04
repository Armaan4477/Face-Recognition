import os
import urllib.request
import bz2

def download_dlib_models():
    """Download required dlib models"""
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    models = {
        "shape_predictor_68_face_landmarks.dat": "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
        "dlib_face_recognition_resnet_model_v1.dat": "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"
    }
    
    for model_name, url in models.items():
        model_path = os.path.join(models_dir, model_name)
        compressed_path = model_path + ".bz2"
        
        if not os.path.exists(model_path):
            print(f"Downloading {model_name}...")
            
            urllib.request.urlretrieve(url, compressed_path)
            
            with bz2.BZ2File(compressed_path, 'rb') as f:
                with open(model_path, 'wb') as out:
                    out.write(f.read())
            
            os.remove(compressed_path)
            print(f"Downloaded and extracted {model_name}")
        else:
            print(f"{model_name} already exists")

if __name__ == "__main__":
    download_dlib_models()
    print("All models downloaded successfully!")
