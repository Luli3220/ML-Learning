'''
Author: LuLi 3048715073@qq.com
Date: 2025-03-04 21:50:18
LastEditors: LuLi 3048715073@qq.com
LastEditTime: 2025-03-07 20:28:54
FilePath: \机器学习-张伟楠\KNN\KNN.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
import matplotlib.pyplot as plt
import os
#导入包
m_x=np.loadtxt('./KNN/mnist_x',delimiter=' ')
m_y=np.loadtxt('./KNN/mnist_y')

data=np.reshape(np.array(m_x[0],dtype=int),[28,28])
plt.figure()
plt.imshow(data,cmap='gray')

#打乱数据
ratio=0.8
split=int(len(m_x)*ratio)

idx=np.random.permutation(np.arange(len(m_x)))
m_x=m_x[idx]
m_y=m_y[idx]

train_x,train_y=m_x[split:],m_y[split:]
test_y,test_y=m_x[:split],m_y[:split]
#定义距离函数
def distance(a,b):
  return np.sqrt(np.sum(np.square(a - b)))  #np.square 对数组每一个进行平方
class KNN:
  

