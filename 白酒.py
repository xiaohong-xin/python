from sklearn.datasets import load_wine
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#白酒
wine=load_wine()
print(wine)

df=pd.DataFrame(wine.data,columns=wine.feature_names)
print(df.head(20))
#拆分数据集
x=wine.data
y=wine.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
print(len(x))
print(len(x_train))
print(len(x_test))
#观察数据
print(df.describe())
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
print("模拟完成")
y_pred=knn.predict(x_test)
print(y_pred[:45])
print(y_test[:45])

accuracy=accuracy_score(y_test,y_pred)
count=sum(y_test==y_pred)
print(accuracy,count)