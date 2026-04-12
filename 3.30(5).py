#分类 类
#回归 值
#线性模型：逻辑回归--->分类
from sklearn.datasets import load_iris
#加载数据
iris=load_iris()
# print(iris)
x=iris.data[:,:2]
y=iris.target
mask=y<2
x=x[mask]
y=y[mask]
# print(len(x))
#第二部 划分训练集和测试集
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
print(len(x_train))
#特征标准化(x-mean)/std
#均值-->0 标准差-->1
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train_scaler=scaler.fit_transform(x_train)
x_test_scaler=scaler.transform(x_test)
# 创建并训练逻辑回归
from sklearn.linear_model import LogisticRegression
#sage 大数据
model=LogisticRegression(C=1.0,solver='lbfgs',max_iter=100,random_state=42)
#训练模型z=wx+wx+wx+....+wx
#权重w     偏执b
model.fit(x_train_scaler,y_train)
print(model.coef_)
print(model.intercept_)
#模型的预测和评估
x_train_pre=model.predict(x_train_scaler)
y_test_pre=model.predict(x_test_scaler)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
train_acc=accuracy_score(y_train,x_train_pre)
test_acc=accuracy_score(y_test,y_test_pre)
print(train_acc,test_acc)

print(classification_report(y_test,y_test_pre))
print(confusion_matrix(y_test,y_test_pre))
