# CV Project 1: Biểu diễn ảnh màu và Lọc tín hiệu

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/broistg/CV-Project-1_Nhom-18/blob/main/notebooks/CV_Project_1_Demo.ipynb)

Bài tập lớn 1 - Computer Vision | HK 2025-2026 | Giảng viên: ThS. Võ Thanh Hùng

## 📖 Giới thiệu

Dự án hiện thực hóa các kỹ thuật xử lý ảnh cơ bản:
- **Biểu diễn ảnh:** Chuyển đổi RGB và Grayscale, tách/gộp các kênh màu
- **Lọc ảnh:** Low-pass (làm trơn), High-pass (tách biên)

Sử dụng **Python** và **OpenCV** (chỉ dùng load ảnh và hỗ trợ phép toán convolution).

## 👥 Thành viên nhóm

| MSSV | Họ và Tên | Công việc thực hiện |
|:---:|:---|:---|
| 2111493 | Nguyễn Minh Khánh | Xử lý ảnh màu |
| 2233163 | Nguyễn Anh Duy | Low-pass filter |
| 2011706 | Nguyễn Nhựt Nguyên | High-pass filter |
| 2310653 | Lê Tiến Đạt | Thực nghiệm & Demo |

## 📂 Cấu trúc thư mục

```
CV-Project-1_Nhom-18/
├── data/
│   ├── input/                  # Ảnh đầu vào
│   └── output/                 # Ảnh kết quả
├── notebooks/
│   └── CV_Project_1_Demo.ipynb # File demo chính
├── src/
│   ├── __init__.py             # Định nghĩa package
│   ├── color_ops.py            # Xử lý màu & kênh
│   ├── spatial_filters.py      # Low-pass & High-pass filters
│   └── utils.py                # Hàm hỗ trợ
├── report/
│   └── CV_Project_title_1_2310653_Lê Tiến Đạt_2233163_Nguyễn Anh Duy_2111493_Nguyễn Minh Khánh_2011706_Nguyễn Nhựt Nguyên.pdf
├── requirements.txt
└── README.md
```

## ⚙️ Cài đặt

**Yêu cầu:** Python 3.x

```bash
# Clone repository
git clone https://github.com/broistg/CV-Project-1_Nhom-18.git
cd CV-Project-1_Nhom-18

# Cài đặt dependencies
pip install -r requirements.txt
```

**Thư viện sử dụng:** numpy, opencv-python, matplotlib

## 🚀 Hướng dẫn chạy

**Cách 1: Google Colab (Khuyên dùng)**
1. Truy cập vào link demo Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/broistg/CV-Project-1_Nhom-18/blob/main/notebooks/CV_Project_1_Demo.ipynb)
2. Nhấn nút "Run all" trong Colab để chạy demo dự án.

**Cách 2: Local**
```bash
jupyter notebook notebooks/CV_Project_1_Demo.ipynb
```

## 📝 Chức năng đã hiện thực

**Phần 1: Biểu diễn ảnh**
- [x] Chuyển đổi RGB và Grayscale
- [x] Tách/gộp các kênh màu
- [x] Hoán đổi màu

**Phần 2: Lọc ảnh**
- [x] Low-pass: Mean Filter, Gaussian Filter
- [x] High-pass: Sobel Filter, Laplacian Filter
- [x] Visualization: So sánh ảnh gốc với sau xử lý

## 🤝 Cam kết

- Code được viết bởi các thành viên nhóm
- Tham khảo tài liệu OpenCV với chú thích rõ ràng
- Không sao chép nguyên văn từ các đồ án khác
