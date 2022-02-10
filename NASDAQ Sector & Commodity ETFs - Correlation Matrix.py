#%%
import yfinance as yf
import pandas as pd 
import numpy as np
import statsmodels.api as sm 
import matplotlib.pyplot as plt
# %%
#List of U.S. ETFs by sectors 
etf_list = ['XLE', 'XLF', 'XLRE', 'XLK', 'XLP', 'XLV', 'XLB', 'XLI', 'XLU', 'XLY', 'XLC', 'GLD', 
 'SLV', 'BNO', 'UGA', 'UNG']
# %%
#Extract data with yf.download
etf = ' '.join(etf_list)
df = yf.download(etf,
                 period = "4y",
                 interval = "1d",
                 group_by = 'ticker',
                 progress = False)
# %%
#Check Columns
df['XLE'].columns
# %%
#Create a new column with returns
for etf in etf_list:
    df[(etf, 'Return')] = df[(etf, 'Close')].pct_change()
# %%
#Create a new DF with returns only
colunas = list()
for dado in etf_list:
    colunas.append((dado, 'Return'))
# %%
df2 = df.filter(items=colunas)
#df2.head()
# %%
#Rename Columns
df2.columns = etf_list
df2
# %%
#Calculate correlation matrix between ETFs
correlation = df2.corr()
# %%
#Plot time 
sm.graphics.plot_corr(correlation, xnames=correlation.columns)
plt.title("Matriz de Correlação")
plt.show()
# %%
