import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# hyperparameters
dim = 10 # dim of input and output vectors
hidden_size = 20 # size of hidden layer of neurons
batch_size = 100
iter_number = 5000
seq_length = 25
learning_rate = 1e-1
step = 0.1 # phase difference between 2 consecutive observations

# model parameters
W1 = np.random.randn(hidden_size, dim)*0.01 # input to hidden
R = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
W2 = np.random.randn(dim, hidden_size)*0.01 # hidden to output
b1 = np.zeros((hidden_size, 1)) # hidden bias
b2 = np.zeros((dim, 1)) # output bias

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def lossFun(inputs, targets, hprev):
	"""
	inputs and targets are both lists of column vectors
	returns the loss and gradients on model parameters
	returns the loss, gradients on model parameters, and last hidden state
	"""
	n = len(inputs)
	z, h, y = {}, {}, {}
	h[-1] = np.copy(hprev)
	loss = 0
	# forward pass
	for t in range(n):
		z[t] = np.dot(W1, inputs[t]) + np.dot(R, h[t-1]) + b1
		h[t] = sigmoid(z[t]) # hidden state
		y[t] = np.dot(W2, h[t]) + b2
		loss += np.linalg.norm(y[t]-targets[t]) # MSE loss
	loss = loss/n
	# backward pass: compute gradients going backwards
	dW1, dW2, dR = np.zeros_like(W1), np.zeros_like(W2), np.zeros_like(R)
	db1, db2 = np.zeros_like(b1), np.zeros_like(b2)
	dhnext = np.zeros_like(h[0])
	for t in reversed(range(n)):
		dy = 2*(y[t]-targets[t]) # backprop into y through loss
		dW2 += np.dot(dy, h[t].T)
		db2 += dy # backprop into W2 and b2
		dh = np.dot(W2.T, dy) + dhnext # backprop into h
		dz = h[t]*(1-h[t])*dh # backprop into z through sigmoid
		dW1 += np.dot(dz, inputs[t].T)
		dR += np.dot(dz, h[t-1].T)
		db1 += dz
		dhnext = np.dot(R.T, dz)
	for dparam in [dW1, dW2, dR, db1, db2]:
		np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
	return loss, dW1, dW2, dR, db1, db2, h[n-1], y

hprev = np.zeros((hidden_size,1))
mW1, mW2, mR = np.zeros_like(W1), np.zeros_like(W2), np.zeros_like(R)
mb1, mb2 = np.zeros_like(b1), np.zeros_like(b2) # memory variables for Adagrad
phases = np.random.rand(dim,1)*2*np.pi
p = 0
pdf = PdfPages('plots.pdf')
for i in range(iter_number):
	# prepare inputs #TODO
	inputs, targets = [], []
	init_phase = np.random.rand()*2*np.pi
	for _ in range(batch_size):
		inputs.append(np.sin(phases+p))
		targets.append(np.sin(phases+p+step))
		p += step
	
	# forward the batch through the net and fetch gradient
	loss, dW1, dW2, dR, db1, db2, hprev, y = lossFun(inputs, targets, hprev)

	# print progress
	if (i+1)%50 == 0: 
		print('iter {}, loss: {:.4f}'.format(i+1, loss)) 

	# plots
	if i+1 in [5,10,20,50,100,500,1000,5000]:
		r = range(batch_size)
		target = [float(targets[i][1]) for i in r]
		output = [float(y[i][1]) for i in r]
		plt.plot(r, target, r, output)
		plt.title('iteration {}'.format(i+1))
		pdf.savefig()
		plt.close()

	# perform parameter update with Adagrad
	for param, dparam, mem in zip([W1, W2, R, b1, b2], 
                                [dW1, dW2, dR, db1, db2], 
                                [mW1, mW2, mR, mb1, mb2]):
		mem += dparam * dparam
		param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

pdf.close()