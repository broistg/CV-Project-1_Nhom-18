import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def read_image(path, mode='rgb'):
    """
    Hàm load ảnh an toàn, kiểm tra file tồn tại.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file ảnh tại: {path}")
    
    if mode == 'gray':
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        # OpenCV đọc BGR, cần chuyển sang RGB
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def normalize_image(image):
    """
    Chuẩn hóa ảnh về dạng hiển thị được (uint8 [0, 255]).
    Thường dùng sau khi thực hiện các phép lọc Sobel/Laplacian (ra float/số âm).
    """
    # Lấy trị tuyệt đối (đối với Sobel/Laplacian biên âm)
    img_abs = np.absolute(image)
    
    # Clip về khoảng 0-255 và chuyển sang uint8
    img_norm = np.clip(img_abs, 0, 255).astype(np.uint8)
    return img_norm

def show_image(image, title="Image", cmap_type='gray'):
    """
    Hàm hiển thị ảnh đơn bằng Matplotlib.
    """
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')

def show_comparison(img1, img2, title1="Original", title2="Processed", cmap1='gray', cmap2='gray'):
    """
    Hiển thị 2 ảnh song song để so sánh trực quan (Before -> After).
    """
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap=cmap1)
    plt.title(title1)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap=cmap2)
    plt.title(title2)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_histogram_comparison(img_orig, img_proc):
    """
    Vẽ biểu đồ Histogram so sánh phân bố cường độ sáng.
    Giúp phân tích độ tương phản hoặc hiệu quả cân bằng sáng.
    """
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(img_orig.ravel(), 256, (0, 256), color='blue', alpha=0.7)
    plt.title('Histogram Ảnh Gốc')
    
    plt.subplot(1, 2, 2)
    plt.hist(img_proc.ravel(), 256, (0, 256), color='red', alpha=0.7)
    plt.title('Histogram Ảnh Sau Xử Lý')
    
    plt.show()