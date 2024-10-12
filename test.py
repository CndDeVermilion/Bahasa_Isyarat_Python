import serial
import time

try:
    # Sesuaikan dengan port Arduino Anda
    ser = serial.Serial('COM5', 9600)
    time.sleep(2)  # Tunggu sebentar agar port serial terbuka dengan benar
    print("Port serial berhasil dibuka.")
except Exception as e:
    print(f"Gagal membuka port serial: {e}")
    exit()

while True:
    try:
        # Kirim data tes ke Arduino
        ser.write(b'T')  # Mengirimkan karakter 'T'
        print("Data dikirim: 'T'")
        time.sleep(1)  # Kirim setiap 1 detik
    except Exception as e:
        print(f"Error saat mengirim data: {e}")
        break
