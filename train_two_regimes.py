# A more complex implementation of HMM for modeling $SPY stock price
# Observable Process: Each output is a day's price movement in 0.5% increments
# Goal: to determine the hidden states underlying the stock market
# Prediction: The two hidden states will represent bull and bear markets

import numpy as np
from hmmlearn import hmm
import yfinance as yf
import matplotlib.pyplot as plt
import joblib

# step 1: get data
spy = yf.Ticker("SPY")
hist = spy.history(start="2000-01-01", end="2023-04-30", interval="1d")
closing_data = hist['Close']
training_data = closing_data[:-20]

output = np.zeros((len(training_data)-1,1), dtype=int)


# output represents half-percentage movements
for i in range(len(training_data)-1):
    temp = round((training_data[i+1]-training_data[i])*100/training_data[i])
    if temp > 5:
        output[i] = 5
    elif temp < -5:
        output[i] = -5
    else:
        output[i] = temp
    output[i] += 5


# step 2: train model
X_train = output[:len(output) // 2]
X_validate = output[len(output) // 2:]

best_score = best_model = None
n_fits = 50
np.random.seed(13)
for idx in range(n_fits):
    model = hmm.CategoricalHMM(n_components=2, random_state=idx, init_params='ste')
    model.fit(X_train)
    score = model.score(X_validate)
    print(f'Model #{idx}\tScore: {score}')
    if best_score is None or score > best_score:
        best_model = model
        best_score = score
print(f'Best score:      {best_score}')

# step 3: View the transition/emission matrices
print("transition matrix:")
print(best_model.transmat_)
print("emission matrix:")
print(best_model.emissionprob_)

# step 3.5: Plot the matrices
fig, ax = plt.subplots()
ax.bar([x for x in range(-5, 6, 1)], best_model.emissionprob_[0])
ax.set_title('State 0 Distribution')
ax.set_xlabel('Percent move')
ax.set_ylabel('Probability')
fig.show()

fig, ax = plt.subplots()
ax.bar([x for x in range(-5, 6, 1)], best_model.emissionprob_[1])
ax.set_title('State 1 Distribution')
ax.set_xlabel('Percent move')
ax.set_ylabel('Probability')
fig.show()

# step 4: Plot the results
real_data = [closing_data[0]]
for i in range(1,len(closing_data)):
    real_data.append(closing_data[i])

model_states = best_model.predict(output)

fig, ax = plt.subplots(2, 1)
ax[0].plot(real_data)#, label='price')
ax[0].set_title('$SPY Price')
ax[0].set_ylabel('Price')

ax[1].plot(model_states) #, label='states')
ax[1].set_ylabel('State')
ax[1].set_xlabel('Days after Jan 1, 2000')
# ax.legend()
plt.show()


# joblib.dump(best_model, "two_regimes_2")

