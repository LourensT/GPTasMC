
# Use shakespeare corpus, 
# reduce to 27 tokens (space and alphabet), and consider context length 4
# Statespace has size 531441.
# %%
with open('./data/full-shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

def reduce_to_consonants_and_values(text):
    
    vowels = 'aeiou'
    consonants = 'bcdfghjklmnpqrstvwxyz'
    # remove capitalization
    text = text.lower()

    tokenized = []
    for c in text:
        if c in vowels:
            tokenized.append(1)
        elif c in consonants:
            tokenized.append(2)
        else: # make space 
            tokenized.append(0)
    
    return tokenized

chars = sorted(list(set(text)))
vocab_size = len(chars)
print('vocab size:', vocab_size, 'unique characters:', ''.join(chars))
# %%
# create a mapping from characters to integers
string_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_string = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [string_to_int[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([int_to_string[i] for i in l]) # decoder: take a list of integers, output a string

import torch 
from torch.nn import functional as F

# seed for reproducibility
torch.manual_seed(0)
# encode the full text
data = torch.tensor(encode(text), dtype=torch.long)
# %%

batch_size = 32
batch_size = 4 # how many independent sequences will we process in parallel?
context_length = 4 # what is the maximum context length for predictions?

def get_batch():
    # generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i:i+context_length] for i in ix])
    y = torch.stack([data[i+context_length] for i in ix])
    return x, y

# import the GPT model architecture and instantiate it
from babyGPT import GPT, GPTConfig

config = GPTConfig(
    block_size = context_length,
    vocab_size = vocab_size,
    n_layer = 4,
    n_head = 4,
    n_embd = 64,
    bias = False,
)
m = GPT(config)
# set optimizer

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

# %%
eval_interval = 100
max_iters = 10000
losses = []
for i in range(max_iters): # increase number of steps for good results... 
    
    # sample a batch of data
    xb, yb = get_batch()

    # evaluate the loss
    logits = m(xb)
    loss = F.cross_entropy(logits, yb) 
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    # every once in a while evaluate the loss on train and val sets
    if i % eval_interval == 0 or i == max_iters - 1:
        print(i, loss.item())
    
    losses.append(loss.item())

print(f"final loss {loss.item()}")
# %%

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))

# %%
# plot loss over time with rolling average
import matplotlib.pyplot as plt
import pandas as pd
plt.plot(pd.Series(losses).rolling(100).mean())
plt.show()

# %%
# get the transition matrix
# get all possible states
import scipy
import itertools
import numpy as np
# state to index
states = [''.join(s) for s in itertools.product(chars, repeat=context_length)]
statesize = len(states)
state_to_index = { s:i for i,s in enumerate(states)}
 
# transition = scipy.sparse.dok_matrix((statesize, statesize))
transition = np.zeros((statesize, statesize))
# %%
for i, state in enumerate(states):
    if i % 1000 == 0:
        print(i)
    state_encoded = encode(state)
    logits = m(torch.tensor(state_encoded, dtype=torch.long)[None, ...])
    for i, rho in enumerate(logits[0]):
        new_state = state[1:] + int_to_string[i]
        transition[state_to_index[state], state_to_index[new_state]] = rho.item()
# %%
# pickle the transition matrix

# csr_transition = transition.tocsr()

import pickle
with open(f'./logits/logits_vocab{vocab_size}context{context_length}.pkl', 'wb') as f:
    pickle.dump(transition, f)

