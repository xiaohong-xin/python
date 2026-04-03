import pandas as pd
data = {
    'student_id': [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010],
    'name': ['张三', '李四', '王五', '赵六', '钱七', '孙八', '周九', '吴十', '郑十一', '王十二'],
    'gender': ['男', '女', '男', '女', '男', None, '男', '女', '男', '女'],  # 有缺失值
    'age': [18, 19, None, 18, 20, 19, 21, 18, None, 19],  # 有缺失值
    'math_score': [85, 92, 78, None, 88, 95, 67, 1000, 82, 90],  # 有缺失值+异常值(1000)
    'english_score': [90, 88, None, 85, 92, 87, 200, 89, 91, 86],  # 有缺失值+异常值(200)
    'admission': [1, 1, 0, 1, 1, 0, 0, 1, 1, None]  # 有缺失值（目标值）
}
df_raw=pd.DataFrame(data)
print(df_raw)
print(df_raw.shape)
print(list(df_raw.columns))
print(df_raw.dtypes)
print(df_raw.head(3))
print(df_raw.tail(3))

print("**"*40)
print(df_raw.describe(include="all"))

import matplotlib.pyplot as plt
import seaborn as sns
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.figure(figsize=(12,6))
# sns.heatmap(df_raw.isnull(),cbar=False,cmap='Reds',yticklabels=False)
# plt.title("缺失值")
# plt.xlabel("列表")
# plt.ylabel("样本")
# plt.tight_layout()
# plt.show()

# fig,axes=plt.subplots(1,2,figsize=(12,5))
# axes[0].boxplot(df_raw['math_score'].dropna())
# axes[0].set_title("数学成绩")
# axes[1].boxplot(df_raw['english_score'].dropna())
# axes[1].set_title("英语成绩")
# plt.show()
from scipy import stats
import numpy as np
math_zscore=np.abs(stats.zscore(df_raw['math_score'].dropna()))
print(math_zscore.round(2))

#IQR
Q1=df_raw['english_score'].quantile(0.25)
Q3=df_raw['english_score'].quantile(0.75)
IQR=Q3-Q1
lowwer_bound=Q1-1.5*IQR
upper_bound=Q3+1.5*IQR
out=df_raw[(df_raw['english_score']>upper_bound | (df_raw['english_score']<lowwer_bound))]
print(out["student_id",'name'])