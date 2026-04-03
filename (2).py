import numpy as np
import pandas as pd
import random
# a=np.array([10,20,30,40,50])
# print(a[0])
# print(a[-1])
# print(a[1:4])
# print(a[:3])
# print(a[::2])
# b=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
# print(b[1,2])
# print(b[0,:])
# print(b[:,1])
# print(b[:2,1:3])
# c=np.array([1,2,3,4,5])
# d=np.array([6,7,8,9,10])
# print(c+d)
# print(c-d)
# print(d**2)
# print(d>8)
#
# row1=np.array([40,400,4000,40000])
# print(b+row1)
#
# data=np.array([[1,2,3],[4,5,6],[7,8,9]])
# print(data.sum())
# print(data.mean())
# print(data.std())
# print(data.min())

# print(data.max())
# print(data.sum(axis=0))
# print(data.sum(axis=1))
# print(data.mean(axis=0))
# print(data.mean(axis=1))

s1=pd.Series([10,20,30,40,50])
print(s1)
s2=pd.Series([10,20,30,40],index=['gaojiajun','gaoruhong','cainengxiang','zhangyuan'])
print(s2)
s3=pd.Series({'sx':90,'xy':89})
print(s3)

data = {
    '姓名': ['张三', '李四', '王五', '赵六'],
    '年龄': [25, 30, 35, 28],
    '城市': ['北京', '上海', '广州', '深圳'],
    '薪资': [15000, 20000, 18000, 22000]
}
print(data)
df2 = pd.DataFrame(np.random.randn(5, 4),
                   columns=['A', 'B', 'C','D'],#row
                   index=['一', '二', '三', '四', '五'])
print(df2)
