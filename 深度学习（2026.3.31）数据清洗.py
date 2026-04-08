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
print(df_raw.describe(include="all"))
print(df_raw.isnull().sum())
#z_score
from scipy import stats
import  numpy as np
math_score=np.abs(stats.zscore(df_raw["math_score"].dropna()))
print(math_score)
Q1=df_raw['english_score'].quantile(0.25)
Q3=df_raw['english_score'].quantile(0.75)
IQR=Q3-Q1
l=Q1-1.5*IQR
u=Q3+1.5*IQR
print(df_raw[(df_raw['english_score']<l)|(df_raw['english_score']>u)])
#众数处理缺失值
df_clean=df_raw.copy()
gender_modo=df_clean['gender'].mode()[0]
print(gender_modo)
df_clean['gender'].fillna(gender_modo,inplace=True)
#均值处理年龄
age_modo=df_clean['age'].mean().round(2)
df_clean['age'].fillna(age_modo,inplace=True)
print(df_clean)

def clean_score(score,max_score):
    if pd.isnull(score):
        return score
    elif score>max_score or score<0:
        return np.nan
    else:
        return score
df_clean['math_score']=df_clean['math_score'].apply(lambda x:clean_score(x,100))
print(df_clean['math_score'].tolist())
df_clean['english_score']=df_clean['english_score'].apply(lambda x:clean_score(x,100))
print(df_clean['english_score'].tolist())

math_media=df_clean['math_score'].median()
df_clean['math_score'].fillna(math_media,inplace=True)
english_media=df_clean['english_score'].median()
df_clean['english_score'].fillna(english_media,inplace=True)
print(df_clean)

df_clean=df_clean.dropna(subset=['admission'])
print(df_clean)
print(df_clean.isnull().sum())

import matplotlib.pyplot as plt
import seaborn as sns
fig,axes=plt.subplots(2,2,figsize=(16,16))
axes[0,0].boxplot([df_raw['math_score'].dropna(),df_clean['math_score']])
axes[0,1].boxplot([df_raw['english_score'].dropna(),df_clean['english_score']])
axes[1,0].boxplot([df_raw['age'].dropna(),df_clean['age']])
plt.tight_layout()
plt.show()
