from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
iris=load_iris()
# print(iris)
df=pd.DataFrame(iris.data,columns=iris.feature_names)
# print(df.head(20))
#拆分数据集
X=iris.data
y=iris.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
print(len(X))
print(len(X_train))
print(len(X_test))

print(df.describe())
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
print("模拟完成")
y_pred=knn.predict(X_test)
print(y_pred[:45])
print(y_test[:45])

accuracy=accuracy_score(y_test,y_pred)
count=sum(y_test==y_pred)
print(accuracy,count)