import numpy as np
import tensorflow as tf

# hyperparameters
dim = 10 # dim of input and output vectors
hidden_size = 20 # size of hidden layer of neurons
batch_size = 100
iter_number = 5000
seq_length = 25
learning_rate = 1e-1
step = 0.1 # phase difference between 2 consecutive observations

# inputs and targets
inputs = tf.placeholder(tf.float32, [None, dim])
targets = tf.placeholder(tf.float32, [None, dim])

# model parameters
W1 = tf.Variable(tf.random_normal([dim, hidden_size], stddev=0.01)) # input to hidden
W2 = tf.Variable(tf.random_normal([hidden_size, dim], stddev=0.01)) # hidden to output
R = tf.Variable(tf.random_normal([hidden_size, hidden_size], stddev=0.01)) # hidden to hidden
b1 = tf.Variable(tf.zeros([hidden_size])) # hidden bias
b2 = tf.Variable(tf.zeros([dim])) # output bias
h = tf.Variable(tf.zeros([1, hidden_size]))

# hidden variables
z = tf.matmul(inputs, W1) + tf.matmul(h, R) + b1
h = tf.nn.sigmoid(z)
y = tf.matmul(h, W2) + b2

# loss and optimizer
mse_loss = tf.losses.mean_squared_error(targets, y)
train_step = tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value=1e-8).minimize(mse_loss)

phases = np.random.rand(dim)*2*np.pi
p = 0

# train
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for i in range(iter_number):
	# prepare inputs
	x, y = [], []
	for _ in range(batch_size):
		x.append(np.sin(phases+p))
		y.append(np.sin(phases+p+step))
		p += step
	inputs_batch = np.vstack(x).astype('float32')
	targets_batch = np.vstack(y).astype('float32')
	# run training step
	_, loss = sess.run([train_step, mse_loss], feed_dict={inputs: inputs_batch, targets: targets_batch})
	# print progress
	if (i+1) % 50 == 0: print('iter {}, loss: {:.8f}'.format(i+1, loss)) 