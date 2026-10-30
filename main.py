import cv2, glob
import numpy as np
import matplotlib.pyplot as plt

CHECKERBOARD = (7, 7)
SQUARE_MM = 18.0   # ganti dengan ukuran kotak hasil ukur
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_MM

objpoints, imgpoints, used = [], [], []


for fname in sorted(glob.glob("foto_ijo/*.jpeg")):
    gray = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print("TIDAK TERBACA:", fname)
        continue

    # deteksi pada versi diperkecil
    scale = min(1.0, 1000 / max(gray.shape))
    small = cv2.resize(gray, None, fx=scale, fy=scale) if scale < 1 else gray

    ret, corners = cv2.findChessboardCorners(small, CHECKERBOARD, flags)
    if not ret:
        print("GAGAL deteksi:", fname)
        continue

    corners = (corners / scale).astype(np.float32)   # kembalikan ke resolusi asli, pastikan float32
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    objpoints.append(objp)
    imgpoints.append(corners2)
    used.append(fname)

# img = cv2.imread("foto_lagi/WhatsApp Image 2026-10-29 at 21.12.40.jpeg")
# idx = used.index("foto_lagi/WhatsApp Image 2026-10-29 at 21.12.40.jpeg")
# vis = cv2.drawChessboardCorners(img, CHECKERBOARD, imgpoints[idx], True)
# cv2.imwrite("cek_5.jpg", vis)

print("Foto valid:", len(used))
if len(used) < 3:
    raise SystemExit("Foto valid kurang dari 3, kalibrasi tidak bisa dijalankan.")

# ===== (1) GANTI pemanggilan calibrateCamera yang lama dengan ini =====
flags = cv2.CALIB_FIX_K3
rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None, flags=flags)

# Mean reprojection error per gambar (dihitung manual)
errs = []
for i in range(len(objpoints)):
    proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
    proj = proj.reshape(-1, 2)
    pts = imgpoints[i].reshape(-1, 2)
    e = np.sqrt(np.mean(np.sum((pts - proj) ** 2, axis=1)))
    errs.append(e)
mean_err = np.mean(errs)

# ===== (2) TAMBAHKAN di sini, SETELAH loop errs selesai =====
for f, e in zip(used, errs):
    print(f"{f}: {e:.2f} px")

print("K =\n", K)
print("D [k1 k2 p1 p2 k3] =", dist.ravel())
print("RMS (OpenCV)      :", rms)
print("Mean reproj error :", mean_err)

# Plot koefisien distorsi
names = ["k1", "k2", "p1", "p2", "k3"]
plt.figure(); plt.bar(names, dist.ravel())
plt.title("Koefisien distorsi"); plt.ylabel("nilai"); plt.grid(axis="y")
plt.savefig("koefisien_distorsi.png", dpi=150)

# Plot error per foto
plt.figure(); plt.bar(range(len(errs)), errs)
plt.axhline(0.5, ls="--", c="r"); plt.xlabel("indeks foto"); plt.ylabel("piksel")
plt.title("Reprojection error per foto")
plt.savefig("error_per_foto.png", dpi=150)