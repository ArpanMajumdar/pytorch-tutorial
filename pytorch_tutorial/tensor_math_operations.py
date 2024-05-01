import torch
from icecream import ic

x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])

# Addition
z1 = torch.empty(3)
torch.add(x, y, out=z1)
ic(z1)

z2 = torch.add(x, y)

z3 = x + y

# Subtraction
z = x - y

# Division
z = torch.true_divide(x, y)
ic(z)

# Inplace operations
t = torch.zeros(3)
t.add_(x)
ic(t)

t += x
ic(t)

# Simple comparison
z = x > 0
ic(z)

# Matrix multiplication
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2)
x3 = x1.mm(x2)
ic(x3)

# Matrix exponentiation
matrix_exp = torch.rand((5, 5))
ic(matrix_exp.matrix_power(3))

# Element-wise multiplication
z = x * y
ic(z)

# Dot product
z = torch.dot(x, y)
ic(z)

# Batch matrix multiplication
batch = 8
n = 10
m = 20
p = 30
t1 = torch.rand((batch, n, m))
t2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(t1, t2)  # (batch, n, p)
ic(out_bmm.shape)
ic(out_bmm)

# Broadcasting
x1 = torch.rand((5, 5))
x2 = torch.rand((1, 5))

z = x1 - x2
ic(z)

# Other useful tensor operations
sum_x = torch.sum(x, dim=0)
values, indices = torch.max(x, dim=0)
values, indices = torch.min(x, dim=0)
abs_x = torch.abs(x)
z = torch.argmax(x, dim=0)
z = torch.argmin(x, dim=0)
mean_x = torch.mean(x.float(), dim=0)
z = torch.eq(x, y)
torch.sort(y, dim=0, descending=False)

z = torch.clamp(x, min=0, max=2)
ic(z)

# Boolean operations
x = torch.tensor([1, 0, 1, 1, 0], dtype=torch.bool)
z = torch.any(x)
ic(z)
z = torch.all(x)
ic(z)
