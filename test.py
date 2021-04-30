import paddle
import numpy as np

tensor = paddle.to_tensor(np.random.randn(5,6),dtype=np.float32)
tmp = paddle.to_tensor(np.random.randn(3,),dtype=np.float32)
print(tmp)
print(tensor[2,3:])
tensor[2,3:] = tmp
print("YES")
