import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

# hyperparameters
dim = 10 # dim of input and output vectors
hidden_size = 20 # size of hidden layer of neurons
batch_size = 128
iter_number = 5000
seq_length = 25
learning_rate = 1e-1
step = 0.1 # phase difference between 2 consecutive observations

# inputs and targets
inputs = tf.placeholder(tf.float32, [None, seq_length, dim])
targets = tf.placeholder(tf.float32, [None, seq_length, dim])

# lstm
lstm_cell = rnn.LSTMCell(hidden_size, num_proj=dim)
outputs, state = rnn.static_rnn(lstm_cell, tf.unstack(inputs, seq_length, 1), dtype=tf.float32)

# loss and optimizer
mse_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(tf.stack(outputs, 1), targets)), 2)) # sum of squares over dim, mean over seq_length and batches
train_step = tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value=1e-8).minimize(mse_loss)

# train
init_phases = np.random.rand(dim)*2*np.pi
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for i in range(iter_number):
	p = np.random.rand(batch_size)*2*np.pi # batch of staring points
	input_list = [np.sin(np.add.outer(p + t*step, init_phases)) for t in range(seq_length)] # list of batch_size x dim
	target_list = [np.sin(np.add.outer(p + (t+1)*step, init_phases)) for t in range(seq_length)]
	input_batch = np.stack(input_list, 1).astype('float32') # batch_size x seq_length x dim
	target_batch = np.stack(target_list, 1).astype('float32')
	sess.run(train_step, feed_dict={inputs: input_batch, targets: target_batch})
	if (i+1) % 100 == 0:
		loss = sess.run(mse_loss, feed_dict={inputs: input_batch, targets: target_batch})
		print('iter {}, loss: {:.8f}'.format(i+1, loss))

# test
test_size = 128
p = np.random.rand(test_size)*2*np.pi
input_list = [np.sin(np.add.outer(p + t*step, init_phases)) for t in range(seq_length)]
target_list = [np.sin(np.add.outer(p + (t+1)*step, init_phases)) for t in range(seq_length)]
input_batch = np.stack(input_list, 1).astype('float32')
target_batch = np.stack(target_list, 1).astype('float32')
test_loss = sess.run(mse_loss, feed_dict={inputs: input_batch, targets: target_batch})
print('Test loss: {:.8f}'.format(test_loss))