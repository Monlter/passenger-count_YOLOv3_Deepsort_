# @Time    : 2020/12/26 18:50
# @Author  : 我丶老陈 
# @FileName: ck_test.py
# @Software: PyCharm
import numpy as np
a = np.array(range(12)).reshape(2,2,3)
print(a)
print("______________________________________")
b = a[..., ::-1]
print(b)