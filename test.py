import numpy as np

# 原始数组
arr = np.array([
    [4, 4, 4, 4, 4, 4, 4, 4, 4],
    [4, 1, 3, 1, 3, 1, 3, 1, 4],
    [4, 3, 1, 3, 1, 3, 1, 1, 4],
    [4, 1, 3, 1, 3, 1, 3, 1, 4],
    [4, 3, 1, 3, 1, 3, 1, 1, 4],
    [4, 1, 3, 1, 3, 1, 3, 1, 4],
    [4, 3, 1, 3, 1, 3, 1, 1, 4],
    [4, 3, 3, 3, 3, 3, 3, 4, 4],
    [4, 4, 4, 4, 4, 4, 4, 4, 4]
])

# 排除倒数第二行和倒数第二列
rows_to_modify = slice(0, -2)  # 所有行，除了倒数第二行
cols_to_modify = slice(0, -2)  # 所有列，除了倒数第二列

# 提取需要修改的部分
sub_arr = arr[rows_to_modify, cols_to_modify]

# 找到数组中为1或3的元素
indices_1_or_3 = np.isin(sub_arr, [1, 3])

# 随机替换1和3
sub_arr[indices_1_or_3] = np.random.choice([1, 3], size=np.sum(indices_1_or_3))

# 将修改后的部分直接赋值回原数组
arr[rows_to_modify, cols_to_modify] = sub_arr

# 打印结果，确保输出格式正确
for row in arr:
    print(f"{list(row)}")
