import numpy as np
import torch
from icecream import ic

device = None
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


my_tensor = torch.tensor(
    [[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device, requires_grad=True
)

ic(my_tensor)
ic(my_tensor.dtype)
ic(my_tensor.device)
ic(my_tensor.shape)

# Other common initialization methods
x = torch.empty((3, 3))
x = torch.zeros((3, 3))
x = torch.rand((3, 3))
x = torch.ones((3, 3))
x = torch.eye(5)
x = torch.arange(start=0, end=1, step=1)
x = torch.linspace(start=0.1, end=1, steps=10)
x = torch.diag(torch.tensor([1, 2, 3]))
ic(x)

# Convert numpy array to tensor and back
np_arr = np.zeros((5, 5))
tensor = torch.from_numpy(np_arr)
ic(tensor)
np_arr_back = tensor.numpy()
ic(np_arr_back)
