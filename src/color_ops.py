import numpy as np
import cv2

def rgb_to_grayscale(image):
    """
    Chuyển đổi ảnh màu RGB sang Grayscale.
    
    Lý thuyết:
    Mắt người nhạy cảm nhất với màu xanh lá (Green), sau đó đến đỏ (Red) và kém nhất với xanh dương (Blue).
    Do đó, công thức tính độ sáng (Luminance) chuẩn là trung bình có trọng số:
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    
    Args:
        image (numpy.ndarray): Ảnh đầu vào chuẩn RGB (H, W, 3).
        
    Returns:
        numpy.ndarray: Ảnh xám (H, W) kiểu uint8.
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Input phải là ảnh màu 3 kênh (RGB)")

    # 1. Tách các kênh màu dưới dạng float để tính toán chính xác
    # Việc tính toán trên float tránh lỗi tràn số (overflow) khi cộng gộp
    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
    
    # 2. Áp dụng công thức Luminance (vector hóa bằng numpy)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    
    # 3. Chuyển về uint8 để hiển thị
    return gray.astype(np.uint8)

def split_channels(image):
    """
    Tách ảnh thành 3 kênh riêng biệt.
    
    Sử dụng cấu trúc mảng 3 chiều của ảnh: Image[H, W, Channel].
    """
    # Kênh 0: Red, Kênh 1: Green, Kênh 2: Blue (nếu ảnh là RGB)
    return image[:, :, 0], image[:, :, 1], image[:, :, 2]

def merge_channels(c1, c2, c3):
    """
    Gộp 3 kênh đơn lẻ thành ảnh màu.
    
    Sử dụng np.dstack (Depth Stack) để chồng các ma trận 2D lên nhau theo chiều sâu.
    """
    return np.dstack((c1, c2, c3))

def grayscale_to_rgb_naive(gray_image):
    """
    Sao chép giá trị xám cho cả 3 kênh R, G, B.
    Minh họa việc tại sao không thể khôi phục màu từ ảnh xám.
    """
    return np.dstack((gray_image, gray_image, gray_image))