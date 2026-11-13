import matplotlib.pyplot as plt
import numpy as np

# 1. TẠO DỮ LIỆU GIẢ LẬP (Diện tích x và Giá nhà y)
# Giả sử giá thực tế có dạng: y = 2*x + 5 + một chút nhiễu
np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # 100 điểm dữ liệu x
y = 4 + 3 * X + np.random.randn(100, 1)  # y thực tế

# 2. KHỞI TẠO THAM SỐ
w = 0.0  # Trọng số ban đầu
b = 0.0  # Bias ban đầu
learning_rate = 0.1  # Tốc độ học alpha
epochs = 100  # Số lần lặp
N = len(X)  # Số lượng mẫu dữ liệu

# 3. THUẬT TOÁN GRADIENT DESCENT (VÒNG LẶP HUẤN LUYỆN)
loss_history = []

for epoch in range(epochs):
    # a. Dự đoán (Forward pass)
    y_pred = w * X + b

    # b. Tính Loss (MSE) để theo dõi
    loss = (1 / (2 * N)) * np.sum((y - y_pred) ** 2)
    loss_history.append(loss)

    # c. Tính đạo hàm (Gradients) theo công thức toán đã chứng minh
    # dw = - (1/N) * sum( X * (y - y_pred) )
    # db = - (1/N) * sum( y - y_pred )
    dw = -(1 / N) * np.sum(X * (y - y_pred))
    db = -(1 / N) * np.sum(y - y_pred)

    # d. Cập nhật w và b
    w = w - learning_rate * dw
    b = b - learning_rate * db

    # In kết quả mỗi 20 vòng lặp
    if (epoch + 1) % 20 == 0:
        print(
            f"Epoch {epoch+1}: Loss = {loss:.4f} | w = {w:.4f} | b = {b:.4f}"
        )

# 4. VẼ ĐỒ THỊ KẾT QUẢ
plt.figure(figsize=(12, 5))

# Đồ thị 1: Đường thẳng dự đoán khớp với dữ liệu
plt.subplot(1, 2, 1)
plt.scatter(X, y, color="blue", label="Dữ liệu thực tế")
plt.plot(X, w * X + b, color="red", linewidth=2, label="Đường thẳng AI học được")
plt.xlabel("Diện tích (X)")
plt.ylabel("Giá nhà (Y)")
plt.title("Mô hình Linear Regression")
plt.legend()

# Đồ thị 2: Độ giảm của Hàm Mất Mát (Loss) qua thời gian
plt.subplot(1, 2, 2)
plt.plot(range(epochs), loss_history, color="green")
plt.xlabel("Số vòng lặp (Epochs)")
plt.ylabel("Loss (MSE)")
plt.title("Hàm Mất Mát giảm dần theo thời gian")

plt.tight_layout()
plt.show()