import torch

# 创建三个四维张量
tensor1 = torch.randn(2, 3, 4, 5)
tensor2 = torch.randn(2, 3, 4, 5)
tensor3 = torch.randn(2, 3, 4, 5)

x = (tensor1, tensor2, tensor3)

print(x.size)

# 沿着第三维拼接
result = torch.cat(x, dim=2)

print(result.shape)
