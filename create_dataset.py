import os
import pickle

import mediapipe as mp
import cv2

# Inisialisasi MediaPipe untuk mendeteksi tangan
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Path direktori data
DATA_DIR = './data'

# Inisialisasi variabel untuk menyimpan data dan label
data = []
labels = []

# Pastikan direktori data ada
if not os.path.exists(DATA_DIR):
    print(f"Error: Direktori {DATA_DIR} tidak ditemukan.")
    exit()

# Loop melalui folder yang ada di DATA_DIR
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue  # Hanya proses folder, skip file yang bukan folder

    # Loop melalui gambar dalam setiap folder
    for img_path in os.listdir(dir_path):
        data_aux = []
        x_ = []
        y_ = []

        img_full_path = os.path.join(dir_path, img_path)
        img = cv2.imread(img_full_path)
        
        if img is None:
            print(f"Error: Gagal membaca gambar {img_full_path}.")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_)
        else:
            print(f"Warning: Tidak ada tangan terdeteksi di gambar {img_full_path}.")

# Simpan data dan label ke file pickle
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data dan label berhasil disimpan dalam file 'data.pickle'.")
