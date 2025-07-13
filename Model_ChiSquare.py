# 卡方检验https://zhuanlan.zhihu.com/p/128905132
import pandas as pd 
import numpy as np 
from scipy import stats
#创建上述表
observed_pd = pd.DataFrame(['1点']*23+['2点']*20+['3点']*18+['4点']*19+['5点']*24+['6点']*16)
expected_pd = pd.DataFrame(['1点']*20+['2点']*20+['3点']*20+['4点']*20+['5点']*20+['6点']*20)
observed_table = pd.crosstab(index=observed_pd[0],columns='count')
expected_table = pd.crosstab(index=expected_pd[0],columns='count')
print(observed_table)
print('——————')
print(expected_table)
#通过公式算出卡方值
observed = observed_table 
expected = expected_table 
chi_squared_stat = ((observed-expected)**2/expected).sum()
print('chi_squared_stat')
print(chi_squared_stat)


crit = stats.chi2.ppf(q=0.95,df=5)  #95置信水平 df = 自由度
print(crit) #临界值，拒绝域的边界 当卡方值大于临界值，则原假设不成立，备择假设成立
P_value = 1-stats.chi2.cdf(x=chi_squared_stat,df=5)
print('P_value')
print(P_value) 

print(stats.chisquare(f_obs=observed, #Array of obversed counts
                f_exp=expected))#Array of expected counts 