# Avoiding naughty states in GPT

A markov-chain approach to finding bounds on the probability of not encountering a given state in continued GPT evaluation.

## Inspired by
"GPT as a finite-state markov chain", [notebook](https://colab.research.google.com/drive/1SiF0KZJp75rUeetKOWqpsA8clmHP6jMg?usp=sharing#scrollTo=mGHwSuHQuTXI) by @karpathy.

## Installation 
Python 3.10

`pip install -r requirements.txt`.

Optional: Install graphviz for visualization.

`sudo apt-get install graphviz` 

## Repository structure

### `model/` 
Contains the GPT implementation, and scripts for training and saving logits.
* `babyGPT.py` - contains very very tiny decoder-only gpt implementation
* `train_GPT.py` - give a string, the vocab and context size, train gpt and save the logits.
* `logits/` contains the trained logits between context-states

### `analysis/`
* `visualization.py` - give some logits and temperature, will visualize the finite state markov chain. 
* `probabability_bound.py` - calculate hitting time and associated hoeffding inequality for markov chains.