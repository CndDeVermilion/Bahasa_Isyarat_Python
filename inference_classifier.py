import pickle
import cv2
import mediapipe as mp
import numpy as np
import time  # Untuk penundaan jika terjadi error

# Muat model yang telah dilatih sebelumnya untuk mengenali gestur
model_dict = pickle.load(open('./model.p', 'rb'))  # File model ada di folder yang sama
model = model_dict['model']

# Coba buka kamera
cap = cv2.VideoCapture(0)  # Menggunakan webcam
if not cap.isOpened():
    print("Error: Kamera tidak terbuka. Pastikan kamera terhubung.")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Label huruf yang akan dikenali
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'I', 4: 'L', 5: 'O', 6: 'U', 7: 'V', 8: 'W', 9: 'Y'}

# Jalankan program terus-menerus, meskipun ada error
while True:
    try:
        data_aux = []
        ret, frame = cap.read()  # Baca frame dari webcam

        # Jika kamera gagal, lewatkan frame ini dan lanjutkan
        if not ret:
            print("Warning: Gagal membaca frame dari kamera. Mencoba lagi...")
            time.sleep(1)  # Tunggu 1 detik sebelum mencoba lagi
            continue

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Proses gestur tangan dengan Mediapipe
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Gambar landmark tangan di frame
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks,  
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Ambil hanya 21 titik landmark (koordinat x dan y)
                for i in range(21):  # 21 titik landmark tangan
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)  # Tambahkan koordinat x
                    data_aux.append(y)  # Tambahkan koordinat y

                # Pastikan jumlah fitur sesuai dengan yang diharapkan model (42 fitur)
                if len(data_aux) == 42:
                    # Prediksi huruf dari gestur tangan
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = labels_dict[int(prediction[0])]

                    # Tampilkan huruf yang dikenali di layar
                    cv2.putText(frame, predicted_character, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Tampilkan frame dengan landmark dan prediksi
        cv2.imshow('frame', frame)

        # Tekan 'q' untuk keluar dari loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        # Tangkap error yang tidak terduga agar program tidak berhenti
        print(f"Error terdeteksi: {e}")
        time.sleep(1)  # Beri jeda waktu untuk mencoba ulang

# Lepaskan resource
cap.release()
cv2.destroyAllWindows()
