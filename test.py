import torch
import torch.nn.functional as F
import numpy as np

a = np.array([ 

[[1,2],[5,7]] ,
[[1,2],[5,7]] ,
[[1,2],[5,7]] 

], dtype=np.float32)

b = torch.from_numpy(a)
print (F.softmax(b, dim=1))