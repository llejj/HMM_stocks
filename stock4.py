# A first implementation of HMM for modeling $SPY stock price
# Observable Process: Each output is a day's price movement (-5,-4,...,+4,+5)

import numpy as np
from hmmlearn import hmm
import yfinance as yf
import matplotlib.pyplot as plt
import joblib

# step 1: get data
spy = yf.Ticker("SPY")
hist = spy.history(start="2000-01-01", end="2023-04-30", interval="1d")

# step 2: load model
loaded_model = joblib.load("percent_model")

# step 3: use model to predict last month's prices
price_changes, gen_states = loaded_model.sample(len(hist['Close']) + 200, random_state=1)

sim_data = [hist['Close'][0]]
for i in range(1,len(hist['Close']) + 200):
    sim_data.append(sim_data[i-1]*(1+(price_changes[i-1][0]-10)/200))

real_data = [hist['Close'][0]]
for i in range(1,len(hist['Close'])):
    real_data.append(hist['Close'][i])


# step 4: view results
fig, ax = plt.subplots()
ax.plot(real_data, label='real_data')
ax.plot(sim_data, label='sim_data')
ax.set_title('$SPY Price')
ax.set_xlabel('days after Jan 1, 2000')
ax.set_ylabel('price')
ax.legend()
plt.show()

