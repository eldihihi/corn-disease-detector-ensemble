# Gunakan base image Python yang stabil
# Python 3.9 seringkali memiliki kompatibilitas TensorFlow yang baik.
FROM python:3.9-slim-buster

# Atur direktori kerja di dalam kontainer
WORKDIR /app

# Salin requirements.txt dan install dependencies terlebih dahulu
# Ini memanfaatkan Docker cache layer jika requirements.txt tidak berubah
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh sisa aplikasi Anda ke dalam direktori kerja
COPY . .

# Tambahkan variabel lingkungan untuk Flask
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Expose port yang digunakan aplikasi Flask (default 5000)
EXPOSE 5000

# Perintah untuk menjalankan aplikasi Flask saat kontainer dimulai
# Pastikan app.py ada di root sesuai struktur yang sudah diperbaiki
CMD ["python", "app.py"]