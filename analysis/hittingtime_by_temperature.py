# %%
from probability_bound import HitT, load_logits
import matplotlib.pyplot as plt
import numpy as np


temperatures = np.linspace(0.1, 1, 1000)
fp_small = "../model/logits/logits_vocab2context3.csv"
transitions_large, states_large = load_logits(fp_small)
fp_large = "../model/logits/logits_vocab3context4.csv"
transitions_small, states_small = load_logits(fp_large)

hitt_small = [HitT(T, transitions_small) for T in temperatures]
hitt_large = [HitT(T, transitions_large) for T in temperatures]
# %%
# plot HitT as a function of temperature
plt.plot(temperatures, hitt_small, label='vocabulary = 2, context = 3')
plt.plot(temperatures, hitt_large, label='vocabulary = 3, context = 4', alpha=0.9)
plt.xlabel("Temperature")
plt.ylabel("HitT")
plt.title("HitT as a function of T")
# plot HitT as a function of temperature
plt.legend()

plt.yscale("log")
# set xlim 
plt.xlim(0, 1)
#plt.ylim(1, 1000)
plt.show()

# %%
temperatures
# %%
