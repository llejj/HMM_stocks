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
training_data = hist['Close'][:-20]

output = np.zeros((len(training_data)-1,1), dtype=int)


# output represents half-percentage movements
for i in range(len(training_data)-1):
    temp = round((training_data[i+1]-training_data[i])*200/training_data[i])
    if temp > 10:
        output[i] = 10
    elif temp < -10:
        output[i] = -10
    else:
        output[i] = temp
    output[i] += 10


# step 2: train model
X_train = output#[:len(output) // 2]
X_validate = output#[len(output) // 2:]

best_score = best_model = None
n_fits = 2
np.random.seed(13)
for idx in range(n_fits):
    model = hmm.CategoricalHMM(n_components=20, random_state=idx, init_params='ste')
    model.fit(X_train)
    score = model.score(X_validate)
    print(f'Model #{idx}\tScore: {score}')
    if best_score is None or score > best_score:
        best_model = model
        best_score = score
print(f'Best score:      {best_score}')

# step 3: use model to predict last month's prices
price_changes, gen_states = best_model.sample(len(hist['Close']) + 200, random_state=1)

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

# step 5: save HMM to file
#joblib.dump(best_model, "percent_model")

