import numpy as np
import tensorflow as tf

# hyperparameters
dim = 10 # dim of input and output vectors
hidden_size = 20 # size of hidden layer of neurons
batch_size = 1 # 50 or 100 for faster training
iter_number = 5000
seq_length = 25
learning_rate = 1e-1
step = 0.1 # phase difference between 2 consecutive observations

# inputs and targets
inputs = tf.placeholder(tf.float32, [None, dim])
targets = tf.placeholder(tf.float32, [None, dim])
h_prev = tf.placeholder(tf.float32, [None, hidden_size])

# model parameters
W1 = tf.Variable(tf.random_normal([dim, hidden_size], stddev=0.01)) # input to hidden
W2 = tf.Variable(tf.random_normal([hidden_size, dim], stddev=0.01)) # hidden to output
R = tf.Variable(tf.random_normal([hidden_size, hidden_size], stddev=0.01)) # hidden to hidden
b1 = tf.Variable(tf.zeros([hidden_size])) # hidden bias
b2 = tf.Variable(tf.zeros([dim])) # output bias


# hidden variables
z = tf.matmul(inputs, W1) + tf.matmul(h_prev, R) + b1
h = tf.nn.sigmoid(z)
y = tf.matmul(h, W2) + b2

# loss and optimizer
mse_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(y, targets)), 1)) # sum of squares over dim, mean over batches
train_step = tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value=1e-8).minimize(mse_loss)

# train
init_phases = np.random.rand(dim)*2*np.pi
state = np.zeros([1, hidden_size]).astype('float32')
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for i in range(iter_number):
	loss_sum = 0
	p = np.random.rand(batch_size)*2*np.pi # batch of staring points
	for t in range(seq_length):
		inputs_batch = np.sin(np.add.outer(p, init_phases)).astype('float32') # shape: batch_size x dim
		targets_batch = np.sin(np.add.outer(p + step, init_phases)).astype('float32')
		feed = {inputs: inputs_batch, targets: targets_batch, h_prev: state}
		_, loss, state = sess.run([train_step, mse_loss, h], feed_dict=feed)
		loss_sum += loss
		p += step
	if (i+1) % 100 == 0: print('iter {}, loss: {:.8f}'.format(i+1, loss_sum/seq_length))