import cv2
import matplotlib.pyplot as plt

img = cv2.imread("anh1.jpg")
if img is None:
    raise ValueError("Không load được ảnh!")

# Load ảnh màu, mặc định openCV dùng định dạng BGR cần chuyển về RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gaussian_3 = cv2.GaussianBlur(img_rgb, (3, 3), sigmaX=1)

#-----------------------------------------------------------------------------
img2 = cv2.imread("anh2.jpg")
if img2 is None:
    raise ValueError("Không load được ảnh!")

img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
gaussian2_3 = cv2.GaussianBlur(img2_rgb, (3, 3), sigmaX=1)



# ===== HIỂN THỊ SO SÁNH =====
plt.figure(figsize=(20, 10))

# Ảnh gốc 1
plt.subplot(2, 2, 1)
plt.imshow(img_rgb)
plt.title("Ảnh gốc 1")
plt.axis("off")

# Ảnh sau khi làm mờ 1
plt.subplot(2, 2, 2)
plt.imshow(gaussian_3)
plt.title("Gaussian Blur 1 (3x3 σ=1)")
plt.axis("off")

# Ảnh gốc 2
plt.subplot(2, 2, 3)
plt.imshow(img2_rgb)
plt.title("Ảnh gốc 2")
plt.axis("off")

# Ảnh sau khi làm mờ 2
plt.subplot(2, 2, 4)
plt.imshow(gaussian2_3)
plt.title("Gaussian Blur 2 (3x3 σ=1)")
plt.axis("off")

plt.tight_layout()
plt.show()
