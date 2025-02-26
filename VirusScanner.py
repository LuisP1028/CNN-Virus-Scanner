import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import tensorflow as tf
import os
import tempfile
import imageio
import pandas as pd
from sklearn.preprocessing import StandardScaler

def extract_image_features(image_path):
    try:
        img = imageio.imread(image_path)
        features = {
            'mean_intensity': np.mean(img),
            'std_intensity': np.std(img),
            'min_intensity': np.min(img),
            'max_intensity': np.max(img)
        }
        return features
    except Exception as e:
        raise ValueError(f"Feature extraction failed: {str(e)}")

class MalwareAnalysisApp:
    def __init__(self, master, model_path, train_csv_path):
        self.master = master
        master.title("Malware Detection System")
        self.model = tf.keras.models.load_model(model_path)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.scaler = StandardScaler()
        self.load_and_fit_scaler(train_csv_path)
        self.create_widgets()

    def load_and_fit_scaler(self, train_csv_path):
        df = pd.read_csv(train_csv_path)
        features = df[['mean_intensity', 'std_intensity', 'min_intensity', 'max_intensity']].values
        self.scaler.fit(features)

    def create_widgets(self):
        self.select_btn = tk.Button(self.master, text="Select File", command=self.analyze_file)
        self.select_btn.pack(pady=20)
        self.result_text = tk.Text(self.master, height=8, width=60)
        self.result_text.pack(pady=10)
        self.result_text.insert(tk.END, "Results will appear here...")
        self.result_text.config(state=tk.DISABLED)

    def analyze_file(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        try:
            img_path = os.path.join(self.temp_dir.name, "temp.png")
            self.file_to_image(file_path, img_path)
            features = extract_image_features(img_path)
            feature_array = np.array([[features['mean_intensity'], features['std_intensity'], features['min_intensity'], features['max_intensity']]])
            feature_array = self.scaler.transform(feature_array)
            prediction = self.model.predict(feature_array)
            confidence = prediction[0][0]
            class_idx = 1 if confidence > 0.5 else 0
            self.display_results(file_path, class_idx)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def file_to_image(self, input_path, output_path):
        try:
            with open(input_path, 'rb') as f:
                file_bytes = f.read()
            target_bytes = 256 * 256
            bytes_len = len(file_bytes)
            if bytes_len < target_bytes:
                padded = file_bytes + bytes([0] * (target_bytes - bytes_len))
                img_data = np.frombuffer(padded, dtype=np.uint8)
            else:
                img_data = np.frombuffer(file_bytes[:target_bytes], dtype=np.uint8)
            img_data = img_data.reshape((256, 256))
            imageio.imwrite(output_path, img_data)
        except Exception as e:
            raise ValueError(f"Image conversion failed: {str(e)}")

    def display_results(self, filename, class_idx):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        classification = "Malware" if class_idx == 1 else "Benign"
        result = (
            f"File: {os.path.basename(filename)}\n"
            f"Classification: {classification}\n"
        )
        self.result_text.insert(tk.END, result)
        self.result_text.config(state=tk.DISABLED)

    def __del__(self):
        self.temp_dir.cleanup()

if __name__ == "__main__":
    MODEL_PATH = r""
    TRAIN_CSV_PATH = r""
    root = tk.Tk()
    app = MalwareAnalysisApp(root, MODEL_PATH, TRAIN_CSV_PATH)
    root.mainloop()

#Edit model path 
#Edit csv path (in malware features)