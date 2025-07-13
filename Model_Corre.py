# https://blog.csdn.net/ddjhpxs/article/details/105767589 相关性分析

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 支持显示中文
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False

data = pd.read_excel('eighth_girl.xlsx')

# 相关系数
print(data.head())
print(data.info())

# 统计信息
dsc = data.describe()
print(dsc)

# 相关系数矩阵图
pd.plotting.scatter_matrix(data, figsize=(20,10), alpha=0.75)
plt.show()

print('============相关系数矩阵============')
cor = data.corr()  # 默认method='pearson'
print(cor)

import seaborn as sns
sns.set(font='SimHei')  # 支持中文显示

fig, ax = plt.subplots(figsize = (10,10))

# cor：相关系数矩阵
# cmap：颜色
# xticklabels：显示x轴标签
# yticklabels：显示y轴标签
# annot：方块中显示数据
# square：方块为正方形

sns.heatmap(cor, cmap='YlGnBu', xticklabels=True, yticklabels=True,
            annot=True, square=True)

plt.show()
# <matplotlib.axes._subplots.AxesSubplot at 0x1c772aa7e88>



from scipy import stats
np.set_printoptions(suppress=True)  # 不使用用科学计数法
pd.set_option('display.float_format',lambda x : '%.4f' % x)  # 保留小数点后4位有效数字
# 0.975分位数
tp = stats.t.isf(1-0.975, 28)

x = np.linspace(-5,5,100)
y = stats.t.pdf(x, 28)
plt.plot(x,y)
plt.vlines(-tp, 0, stats.t.pdf(-tp, 28), colors='orange')
plt.vlines(tp, 0, stats.t.pdf(tp, 28), colors='orange')
plt.fill_between(x, 0, y, where=abs(x)>tp, interpolate=True, color='r')

plt.show()


# 自定义求解p值矩阵的函数
def my_pvalue_pearson(x):
    col = x.shape[1]
    col_name = x.columns.values
    p_val = []
    for i in range(col):
        for j in range(col):
            p_val.append(stats.pearsonr(x[col_name[i]], x[col_name[j]])[1])
    p_val = pd.DataFrame(np.array(p_val).reshape(col, col), columns=col_name, index=col_name)
    # p_val.to_csv('p_val_pearson.csv')  # 此处实则为多此一举，目的是借助带有excel格式的数据使得输出更美观
    # p_val = pd.read_csv('p_val_pearson.csv', index_col=0)
    return p_val

print('p值')
print(my_pvalue_pearson(data))

x = stats.norm.rvs(2, 3, 100)    
skewness = stats.skew(x)  # 偏度
kurtosis = stats.kurtosis(x)  # 峰度
jbtext = stats.jarque_bera(x)
print('偏度为：',skewness)
print('峰度为：',kurtosis)
print('J-B值：',jbtext[0])
print('p-value:',jbtext[1])


def my_jbtext(x):
    col_name = x.columns.values
    col_cnt = x.shape[1]
    h_mat = np.zeros(col_cnt)
    p_mat = np.zeros(col_cnt)
    for i in range(col_cnt):
        p_val = stats.jarque_bera(data[col_name[i]])[1]
        p_mat[i] = p_val
        if p_val >= 0.05:
            h_mat[i] = 0  # 通过原假设
        else:
            h_mat[i] = 1  # 拒绝原假设
    print(h_mat)
    print(p_mat)  # 各列的p值

my_jbtext(data)

print(stats.shapiro(data['身高']))  # 单个变量

def my_shaptext(x):
    col_name = x.columns.values
    col_cnt = x.shape[1]
    h_mat = np.zeros(col_cnt)
    p_mat = np.zeros(col_cnt)
    for i in range(col_cnt):
        p_val = stats.shapiro(data[col_name[i]])[1]
        p_mat[i] = p_val
        if p_val >= 0.05:
            h_mat[i] = 0  # 通过原假设
        else:
            h_mat[i] = 1  # 拒绝原假设
    print(h_mat)
    print(p_mat)  # 各列的p值

my_shaptext(data)


stats.probplot(data['身高'], dist="norm", plot=plt)
plt.show()

data.corr(method='spearman')

# 自定义求解p值矩阵的函数
def my_pvalue_spearman(x):
    col = x.shape[1]
    col_name = x.columns.values
    p_val = []
    for i in range(col):
        for j in range(col):
            p_val.append(stats.spearmanr(x[col_name[i]], x[col_name[j]])[1])
    p_val = pd.DataFrame(np.array(p_val).reshape(col, col), columns=col_name, index=col_name)
    # p_val.to_csv('p_val_spearman.csv')  # 此处实则为多此一举，目的是借助带有excel格式的数据使得输出更美观
    # p_val = pd.read_csv('p_val_spearman.csv', index_col=0)
    return p_val

print(my_pvalue_spearman(data))
