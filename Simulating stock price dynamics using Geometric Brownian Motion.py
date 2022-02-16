#%%
#import libraries 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import datetime as dt
import yfinance as yf
plt.style.use('dark_background')
# %%
#Define the parameters for downloading data 

stock = 'AAPL'
start_date = '2018-01-01'
end_date = '2022-02-15'

df = yf.download(stock,
                 start = start_date,
                 end = end_date,
                 group_by = 'ticker',
                 ajusted = True)

df.head()
# %%
#Create a new column with returns

df['Return'] = df['Adj Close'].pct_change().dropna()
df.head()
# %%
#Create a new Series with returns only

adj_close = df['Adj Close']
df2 = df['Return'].dropna()
df2.head()
# %%
# Plote the daily returns

print(f'Average return {stock}: {100 * df2.mean():.2f}%')
plt.figure(figsize=(21, 7))
df2.plot(title=f' {stock} returns: {start_date} - {end_date}');
# %%
# Split the data into training and test sets 

train = df2['2019-01-01':'2019-12-31']
test = df2['2020-01-01':'2020-01-31']
# %%
# Specify the parameters of the simulation (T, N, S_0, N_SIM, mu, sigma)

T = len(test)
N = len(test)
S_0 = adj_close[train.index[-1]]
N_SIM = 300
mu = train.mean()
sigma = train.std()
# %%
# Define the function for simulations 

def simulate_gbm(s_0, mu, sigma, n_sims, T, N):
 dt = T/N
 dW = np.random.normal(scale = np.sqrt(dt),
 size=(n_sims, N))
 W = np.cumsum(dW, axis=1)
 time_step = np.linspace(dt, T, N)
 time_steps = np.broadcast_to(time_step, (n_sims, N))
 S_t = s_0 * np.exp((mu - 0.5 * sigma ** 2) * time_steps
 + sigma * W)
 S_t = np.insert(S_t, 0, s_0, axis=1)
 return S_t
# %%
# Run the simulations 

gbm_simulations = simulate_gbm(S_0, mu, sigma, N_SIM, T, N)
# %%
#Prepare objects for plotting

LAST_TRAIN_DATE = train.index[-1].date()
FIRST_TEST_DATE = test.index[0].date()
LAST_TEST_DATE = test.index[-1].date()
PLOT_TITLE = (f'{stock} Simulation '
 f'({FIRST_TEST_DATE}:{LAST_TEST_DATE})')
selected_indices = adj_close[LAST_TRAIN_DATE:LAST_TEST_DATE].index
index = [date.date() for date in selected_indices]
gbm_simulations_df = pd.DataFrame(np.transpose(gbm_simulations),
 index=index)
# %%
# plotting

ax = gbm_simulations_df.plot(alpha=0.2, legend=False)
line_1, = ax.plot(index, gbm_simulations_df.mean(axis=1),
 color='red')
line_2, = ax.plot(index, adj_close[LAST_TRAIN_DATE:LAST_TEST_DATE],
 color='blue')
ax.set_title(PLOT_TITLE, fontsize=16)
ax.legend((line_1, line_2), ('mean', 'actual'));
# %%
