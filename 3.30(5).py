from sklearn.datasets import load_iris
iris=load_iris()
# print(iris)
x=iris.data[:,:2]
y=iris.target
mask=y<2
x=x[mask]
y=y[mask]
# print(len(x))

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
print(len(x_train))

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train_scaler=scaler.fit_transform(x_train)
x_test_scaler=scaler.transform(x_test)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression(C=1.0,solver='lbfgs',max_iter=100,random_state=42)
model.fit(x_train_scaler,y_train)
print(model.coef_)
print(model.intercept_)

x_train_pre=model.predict(x_train_scaler)
y_test_pre=model.predict(x_test_scaler)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
train_acc=accuracy_score(y_train,x_train_pre)
test_acc=accuracy_score(y_test,y_test_pre)
print(train_acc,test_acc)

print(classification_report(y_test,y_test_pre))
print(confusion_matrix(y_test,y_test_pre))
