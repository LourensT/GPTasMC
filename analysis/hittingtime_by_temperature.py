# %%
from probability_bound import HitT, load_logits
import matplotlib.pyplot as plt
import numpy as np

temperatures = np.linspace(0.1, 1, 100)
fp_small = "../model/logits/logits_vocab2context3.csv"
transitions_small, states_small = load_logits(fp_small)

fp_large = "../model/logits/logits_vocab3context4.csv"
transitions_large, states_large = load_logits(fp_large)
# %%
hitt_small = [HitT(t, transitions_small) for t in temperatures]
hitt_large = [HitT(t, transitions_large) for t in temperatures]
# %%
# plot HitT as a function of temperature
# set dpi
plt.figure(dpi=300)
plt.plot(temperatures, hitt_small, label='vocabulary = 2, context = 3')
plt.plot(temperatures, hitt_large, label='vocabulary = 3, context = 4', alpha=0.9)
plt.xlabel("Temperature")
plt.ylabel("HitT")
plt.title("HitT as a function of T")
# plot HitT as a function of temperature
plt.legend()
plt.yscale("log")
# set xlim
#plt.ylim(-0.01*max(hitt_large), max(hitt_large))
#plt.xlim(0.0, 0.15)
plt.show()

# %%
temperatures
# %%
hitt_large[list(temperatures).index(0.6)]
# %%
temperatures
# %%
HitT(0.6, transitions_large)
# %%
hitt_large
# %%
len(temperatures)
list(temperatures).index(0.6)
# %%
HitT(0.6, transitions_large)
# %%
