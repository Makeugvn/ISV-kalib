import cv2

img = cv2.imread("foto_lagi/WhatsApp Image 2026-10-29 at 21.12.40.jpeg")
print("Ukuran foto:", None if img is None else img.shape)   # None = path salah

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# perkecil jika foto sangat besar
scale = 1000 / max(gray.shape)
if scale < 1:
    gray = cv2.resize(gray, None, fx=scale, fy=scale)

flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
         + cv2.CALIB_CB_NORMALIZE_IMAGE
         + cv2.CALIB_CB_FAST_CHECK * 0)

for cols in range(4, 13):
    for rows in range(3, 11):
        if cols < rows:
            continue
        ok, _ = cv2.findChessboardCorners(gray, (cols, rows), flags)
        if ok:
            print("TERDETEKSI dengan ukuran:", (cols, rows))


