# A first implementation of HMM for modeling $SPY stock price
# Observable Process: Each output is a day's price movement (-5,-4,...,+4,+5)

import numpy as np
from hmmlearn import hmm
import yfinance as yf
import matplotlib.pyplot as plt

# step 1: get data
spy = yf.Ticker("SPY")
hist = spy.history(start="2000-01-01", end="2023-04-30", interval="1d")
training_data = hist['Close'][:-20]

output = np.zeros((len(training_data)-1,1), dtype=int)

for i in range(len(training_data)-1):
    output[i] = round(training_data[i+1]-training_data[i])
    if output[i] > 5:
        output[i] = 5
    if output[i] < -5:
        output[i] = -5
    output[i] += 5

# step 2: train model
X_train = output#[:len(output) // 2]
X_validate = output#[len(output) // 2:]

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

# step 3: use model to predict last month's prices
price_changes, gen_states = best_model.sample(len(hist['Close']))

real_data = [hist['Close'][0]]
sim_data = [hist['Close'][0]]
for i in range(1,len(hist['Close'])):
    sim_data.append(sim_data[i-1] + price_changes[i-1][0] - 5)
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



"""
# step 4: view results
fig, ax = plt.subplots()
ax.plot(output[:500], label='actual')
ax.plot(price_changes[:500] + 11.5, label='simulated')
ax.set_yticks([])
ax.set_title('daily price changes')
ax.set_xlabel('Time (# days)')
plt.show()
"""