import numpy as np
from display_network import *
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans

# 1. Dùng cách đã chạy thành công ở unzip.py để lấy dữ liệu
print("Đang tải dữ liệu...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')

# 2. Lấy 1000 ảnh đầu tiên và chuẩn hóa (giống hệt logic X0 cũ của bạn)
X = mnist.data[:1000] / 256.0

# 3. Chạy thuật toán KMeans
K = 10
kmeans = KMeans(n_clusters=K).fit(X)
pred_label = kmeans.predict(X)

print("KMeans chạy thành công!")