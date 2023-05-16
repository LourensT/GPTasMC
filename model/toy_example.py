# Example with toy-data, considering two tokens (0,1) in context length 3. 
# %%
from babyGPT import GPT, GPTConfig
import torch
from torch.nn import functional as F

context_length = 3
vocab_size = 2

config = GPTConfig(
    block_size = context_length,
    vocab_size = vocab_size,
    n_layer = 4,
    n_head = 4,
    n_embd = 16,
    bias = False,
)

gpt = GPT(config)
seq = list(map(int, "111101111011110"))
# %%
# convert the sequence to a tensor
#  holding all the individual examples in that sequence
X, Y = [], []
# iterate over the sequence and grab every consecutive 3 bits
# the correct label for what's next is the next bit at each position
for i in range(len(seq) - context_length):
    X.append(seq[i:i+context_length])
    Y.append(seq[i+context_length])

X = torch.tensor(X, dtype=torch.long)
Y = torch.tensor(Y, dtype=torch.long)
print(X.shape, Y.shape)

# %%
torch.manual_seed(0)
gpt = GPT(config)
optimizer = torch.optim.AdamW(gpt.parameters(), lr=1e-3, weight_decay=1e-1)

# train the GPT for some number of iterations
for i in range(1000):
    logits = gpt(X)
    loss = F.cross_entropy(logits, Y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(i, loss.item())

def all_possible(n, k):
    # return all possible lists of k elements, each in range of [0,n)
    if k == 0:
        yield []
    else:
        for i in range(n):
            for c in all_possible(n, k - 1):
                yield [i] + c

states = list(all_possible(vocab_size, context_length))

# get transition logits for each possible next state
# and save them in a matrix
transitions = torch.zeros((len(states), len(states)))
for i, state in enumerate(states):
    x = torch.tensor(state, dtype=torch.long)[None, ...] # turn the list into a torch tensor and add a batch dimension
    logits = gpt(x) # forward pass

    new_state_0 = state[1:] + [0]
    j = states.index(new_state_0)
    transitions[i, j] = logits[0, 0] # the logit for transitioning to new_state_0

    new_state_1 = state[1:] + [1]
    j = states.index(new_state_1)
    transitions[i, j] = logits[0, 1] # the logit for transitioning to new_state_1

# save transition matrix to a csv
with open(f"logits_vocab{vocab_size}context{context_length}.csv", "w") as f:
    # write states as header
    for i, state in enumerate(states):
        if i > 0:
            f.write(",")
        f.write("".join(map(str, state)))
        
    f.write("\n")

    for row in transitions:
        for i, cell in enumerate(row):
            if i > 0:
                f.write(",")
            f.write(str(cell.item()))
        f.write("\n")