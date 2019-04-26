import torch
from torch import Tensor

B = 10 # batch size
F = 7 # num features
G = 6
N = 100 # num agents
K = 3 # num aggregations

# Deep Aggregation GNN with no delay:

# Input: 
# Current state: x_t of shape (B,1,F,N), where B = # batches, F = # features, N = # agents
# Current GSO: GSO = [I, A_t, A_t^2,..., A_t^K-1] of shape (B,K,N,N)

hidden_layers = [7,6,5,4]

# apply the GSO - done
inp = torch.ones((B,1,F,N)) # batch x features x agents
GSO = torch.ones((B,K,N,N)) # a different KxNxN GSO for each state in the batch

for i in range(len(hidden_layers)-1):

	agg = torch.matmul(inp,GSO) # output is batch x K x F x N
	print(agg.size())
	m = torch.nn.Conv2d(K, hidden_layers[i+1], kernel_size=(hidden_layers[i], 1), stride=(hidden_layers[i], 1))
	out = m(agg)
	out = torch.nn.functional.relu(out)
	inp = out.view((-1,1,hidden_layers[i+1],N))
	print(inp.size())

#####################################################################
# Delayed Aggregation GNN

print('Delayed')
# Input: 
# History of states: x_t, x_t-1,...x_t-k+1 of shape (B,K,F,N)
# Delayed GSO: [I, A_t, A_t A_t-1, A_t ... A_t-k+1] of shape (B,K,N,N)

inp = torch.ones((B,K,F,N)) 
GSO = torch.ones((B,K,N,N))

# swap K and F dimensions
inp = inp.permute(0,2,1,3)
m = torch.nn.Conv2d(F, G, kernel_size=(1, 1), stride=(1, 1))
out = m(inp)
print(out.size())

# swap back for the multiplication to work
out = out.permute(0,2,1,3)
agg = torch.matmul(out,GSO) 
print(agg.size())

# use same conv1d trick for linear weights



