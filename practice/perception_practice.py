import tensorflow as tf 
import numpy as np 
from scipy.misc import imread, imresize

import random

import sys
sys.path.insert(0,'../')

import Perception


lr = 4e-4
n_epochs = 1000
d_size = 100
b_size = 20
n_batches = d_size // b_size

ht = 84
wd = 84

state_size = 512

action_space = 3 # 0: switch rg, 1: switch rb, 2: switch gb


def get_imgs():
	d = []
	data_dir = 'data/{0}.jpg'
	i = 1
	while(len(d) < d_size):
		try:
			im = imread(data_dir.format(i))
			d.append(imresize(im,(ht,wd)))
			i += 1
		except:
			i += 1

	d = np.array(d,dtype=np.float32)/255

	return d

d = get_imgs()




X = tf.placeholder(tf.float32,shape=[None,ht,wd,3])
Y = tf.placeholder(tf.float32,shape=[None,ht,wd,3])
actions = tf.placeholder(tf.float32,shape=[None,action_space])


hidden_state = Perception.CNN(X,ht,wd,3,state_size,'net')

Yhat = Perception.Predictor(hidden_state, state_size,
					actions,ht,wd,
					out_channel=3,
					network_name='dnet')



error = tf.reduce_mean(tf.square(Y-Yhat))

optimizer = tf.train.AdamOptimizer(lr)
train_step = optimizer.minimize(error)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


#w = 1/0

for epoch in range(n_epochs):
	mean_loss = 0
	for b in range(n_batches):

		x = d[b*b_size:(b+1)*b_size]
		a = random.choice([0,1,2])
		ain = np.zeros((b_size,action_space))
		ain[:,a] = 1
		y = np.copy(x)
		if a == 0:
			y[:,:,:,0] = x[:,:,:,1]
			y[:,:,:,1] = x[:,:,:,0]
		elif a == 1:
			y[:,:,:,0] = x[:,:,:,2]
			y[:,:,:,2] = x[:,:,:,0]
		elif a == 2:
			y[:,:,:,1] = x[:,:,:,2]
			y[:,:,:,2] = x[:,:,:,1]

		_,l = sess.run([train_step,error],feed_dict={X:x,
						Y:y,actions:ain})
		mean_loss += l.mean()/n_batches

	print('epoch {0} | mean_loss = {1}'.format(epoch,mean_loss))

















