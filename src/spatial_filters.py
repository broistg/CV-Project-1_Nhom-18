import numpy as np
import cv2

# --- HÀM HỖ TRỢ ---
def create_gaussian_kernel(size, sigma):
    """
    Sinh Gaussian Kernel từ công thức toán học.
    
    Công thức: G(x, y) = (1 / (2*pi*sigma^2)) * exp(-(x^2 + y^2) / (2*sigma^2))
    
    Args:
        size (int): Kích thước kernel (phải là số lẻ, v.d: 3, 5, 7).
        sigma (float): Độ lệch chuẩn, kiểm soát độ mờ.
        
    Returns:
        numpy.ndarray: Ma trận kernel chuẩn hóa (tổng = 1).
    """
    if size % 2 == 0:
        raise ValueError("Kernel size phải là số lẻ.")
        
    # 1. Tạo lưới tọa độ (x, y) với tâm là (0, 0)
    # Ví dụ size=3 -> ax = [-1, 0, 1]
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)
    
    # 2. Áp dụng công thức Gaussian
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    
    # 3. Chuẩn hóa để tổng các phần tử trong kernel = 1
    # Điều này đảm bảo độ sáng trung bình của ảnh không đổi sau khi lọc
    return kernel / np.sum(kernel)

def apply_convolution(image, kernel):
    """
    Wrapper gọi hàm cv2.filter2D.
    Sử dụng ddepth=-1 để giữ nguyên kiểu dữ liệu (nếu input uint8 -> output uint8),
    hoặc cv2.CV_64F nếu cần độ chính xác cao cho số âm (Sobel/Laplacian).
    """
    # Với các kernel có hệ số âm (Sobel, Laplacian), ta cần giữ giá trị thực
    if np.any(kernel < 0):
        return cv2.filter2D(image, cv2.CV_64F, kernel)
    
    # Với kernel dương toàn bộ (Mean, Gaussian), trả về cùng kiểu với ảnh gốc
    return cv2.filter2D(image, -1, kernel)

# --- LOW-PASS FILTERS (Làm trơn) ---
def apply_mean_filter(image, kernel_size):
    """
    Tạo Mean Kernel thủ công: Tất cả giá trị đều là 1/(size^2).
    """
    # Tạo ma trận toàn số 1 chia cho tổng diện tích kernel
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
    return apply_convolution(image, kernel)

def apply_gaussian_filter(image, kernel_size, sigma=1.0):
    """
    Áp dụng lọc Gaussian với kernel tự sinh.
    """
    # Nếu sigma=0, tự tính sigma dựa theo kích thước kernel (giống OpenCV)
    if sigma == 0:
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        
    kernel = create_gaussian_kernel(kernel_size, sigma)
    return apply_convolution(image, kernel)

def apply_median_filter(image, kernel_size):
    """
    Lọc trung vị 
    """
    if kernel_size % 2 == 0:
        kernel_size += 1 # Bắt buộc số lẻ
        
    h, w = image.shape
    pad = kernel_size // 2
    
    # 1. Padding: Dùng reflect để biên ảnh tự nhiên
    img_padded = np.pad(image, ((pad, pad), (pad, pad)), mode='reflect')
    
    # 2. Tạo ảnh output rỗng
    output = np.zeros_like(image)
    
    # 3. Lọc trung vị
    for i in range(h):
        for j in range(w):
            window = img_padded[i:i+kernel_size, j:j+kernel_size]
            output[i, j] = np.median(window)
    return output

# --- HIGH-PASS FILTERS (Tách biên) ---
def apply_sobel(image, orient='x'):
    """
    Sobel Operator.
    Trả về ảnh float64 vì có giá trị âm.
    """
    if orient == 'x':
        # Kernel Sobel X thủ công
        kernel = np.array([[-1, 0, 1], 
                           [-2, 0, 2], 
                           [-1, 0, 1]], dtype=np.float32)
    else:
        # Kernel Sobel Y thủ công
        kernel = np.array([[-1, -2, -1], 
                           [0, 0, 0], 
                           [1, 2, 1]], dtype=np.float32)
        
    return apply_convolution(image, kernel)

def apply_laplacian(image, method='standard'):
    """
    Laplacian Operator với nhiều biến thể kernel khác nhau.
    - 'standard': 4-neighbor (Cơ bản).
    - 'enhanced': 8-neighbor (Bắt biên mạnh hơn, gồm cả đường chéo).
    - 'log': Laplacian of Gaussian 5x5 (Giảm nhiễu tốt nhất).
    """
    if method == 'standard':
        # 4-neighbor: Đạo hàm bậc 2 theo X và Y
        kernel = np.array([[0, -1, 0], 
                           [-1, 4, -1], 
                           [0, -1, 0]], dtype=np.float32)
        
    elif method == 'enhanced':
        # 8-neighbor: Bao gồm cả đường chéo -> Đẳng hướng (Isotropic) tốt hơn
        kernel = np.array([[-1, -1, -1], 
                           [-1, 8, -1], 
                           [-1, -1, -1]], dtype=np.float32)
        
    elif method == 'log':
        # LoG (Laplacian of Gaussian) 5x5 approximation
        # Đây là xấp xỉ của hàm "Mexican Hat"
        # Giúp làm trơn nhiễu trước khi tính đạo hàm
        kernel = np.array([[0,  0, -1,  0,  0],
                           [0, -1, -2, -1,  0],
                           [-1, -2, 16, -2, -1],
                           [0, -1, -2, -1,  0],
                           [0,  0, -1,  0,  0]], dtype=np.float32)
    else:
        raise ValueError("Method phải là 'standard', 'enhanced' hoặc 'log'")

    return apply_convolution(image, kernel)

def apply_gaussian_highpass(image, kernel_size=5, sigma=1.0):
    """
    Sử dụng Gaussian để làm High-pass (Unsharp Masking technique).
    Công thức: HighPass = Image - LowPass(Image)
    """
    # 1. Tạo ảnh Low-pass (bị làm mờ)
    blurred = apply_gaussian_filter(image, kernel_size, sigma)
    
    # 2. Chuyển sang float để trừ không bị lỗi
    img_float = image.astype(np.float32)
    blur_float = blurred.astype(np.float32)
    
    # 3. Trừ ảnh gốc cho ảnh mờ -> Chỉ còn lại các chi tiết cạnh
    high_pass = img_float - blur_float
    
    # Trả về kết quả (có thể có số âm)
    return high_pass