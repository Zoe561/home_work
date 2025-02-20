import torch

# 任務 1：從原始 Python 列表建立 Tensor，並列印形狀與數據類型
list_data = [[2, 3, 1], 
             [5, -2, 1]]
tensor_data = torch.tensor(list_data)  # 將列表轉為 Tensor
print("任務 1：Tensor 內容：\n", tensor_data)
print("任務 1：Tensor 形狀（shape）：", tensor_data.shape)
print("任務 1：Tensor 數據類型（dtype）：", tensor_data.dtype)
print("-" * 50)

# 任務 2：建立 3×4×2 的隨機 Tensor（0 ~ 1 之間），並列印形狀與數值
random_tensor = torch.rand((3, 4, 2))  # rand 會產生 [0, 1) 區間的隨機數
print("任務 2：隨機 Tensor 內容：\n", random_tensor)
print("任務 2：隨機 Tensor 形狀（shape）：", random_tensor.shape)
print("-" * 50)

# 任務 3：建立 2×1×5 的全為 1 的 Tensor，並列印形狀與數值
ones_tensor = torch.ones((2, 1, 5))
print("任務 3：全為 1 的 Tensor 內容：\n", ones_tensor)
print("任務 3：Tensor 形狀（shape）：", ones_tensor.shape)
print("-" * 50)

# 任務 4：矩陣相乘
matrix_A = torch.tensor([[1, 2, 4], 
                         [2, 1, 3]])
matrix_B = torch.tensor([[5], 
                         [2], 
                         [1]])
# 矩陣相乘可使用 '@' 或 torch.matmul
matmul_result = matrix_A @ matrix_B
print("任務 4：矩陣相乘結果：\n", matmul_result)
print("-" * 50)

# 任務 5：元素對應相乘（逐元素相乘）
element_A = torch.tensor([[1, 2], 
                          [2, 3],
                          [-1, 3]])
element_B = torch.tensor([[5,  4], 
                          [2,  1], 
                          [1, -5]])
elementwise_result = element_A * element_B
print("任務 5：逐元素相乘結果：\n", elementwise_result)
