# A first implementation of HMM for modeling $SPY stock price
# Observable Process: Each output is a day's price movement (up,down,flat)

import numpy as np
from hmmlearn import hmm
import yfinance as yf
import matplotlib.pyplot as plt

# step 1: get data
spy = yf.Ticker("SPY")
hist = spy.history(start="2000-01-01", end="2023-03-31", interval="1d")
data = hist['Close']

output = np.zeros(((len(data)-1),1), dtype=int)

for i in range(len(data)-1):
    output[i] = round(data[i+1]-data[i])
    if output[i] > 2:
        output[i] = 1
    elif output[i] < -2:
        output[i] = -1
    else:
        output[i] = 0
    output[i] += 1

# step 2: train model
X_train = output[:len(output) - 500]
X_validate = output[len(output) - 500:]

best_score = best_model = None
n_fits = 20
np.random.seed(13)
for idx in range(n_fits):
    model = hmm.CategoricalHMM(n_components=10, random_state=idx, init_params='ste')
    model.fit(X_train)
    score = model.score(X_validate)
    print(f'Model #{idx}\tScore: {score}')
    if best_score is None or score > best_score:
        best_model = model
        best_score = score

print(f'Best score:      {best_score}')

price_changes, gen_states = best_model.sample(len(output))

# step 3: use model to predict last month's prices
fig, ax = plt.subplots()
ax.plot(output[:2000], label='actual')
ax.plot(price_changes[:2000] + 3, label='simulated')
ax.set_yticks([])
ax.set_title('daily price changes')
ax.set_xlabel('Time (# days)')
plt.show()
