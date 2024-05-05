import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

from utils.dataGenerator import extract_keypoints

image_path = r"C:\Users\rrsan\Documents\My Docs\College\Projects\StaircaseFall-detection\data\Images\fall\1.jpg"

# Load the model
with open('models/model.pkl', 'rb') as file:
    model = pickle.load(file)

keypoints = extract_keypoints(image_path)
row = []
if keypoints:
    for kp in keypoints:
        row.extend(kp)

print(row)

row_array = np.array(row)
row = row_array.reshape(1, -1)

scaler = StandardScaler()
row_scaled = scaler.fit_transform(row)

pred = model.predict(row_scaled)
print(pred)

