
import tensorflow as tf 
from tensorflow.contrib import slim
import numpy as np 
from tensorflow.contrib.layers import xavier_initializer

conv3_shape_1 = None
conv3_shape_2 = None
conv3_shape_3 = None

conv2_shape_1 = None
conv2_shape_2 = None
conv2_shape_3 = None

conv1_shape_1 = None
conv1_shape_2 = None
conv1_shape_3 = None

seed = 1234

def CNN(input, height, width,in_channel, out_channel,afn=tf.nn.elu,weights=[]):

	img = tf.reshape(input, shape=[-1, height, width, in_channel])
	# filter [filter ht, filter wd, inchannels, outchannels]
	if len(weights) == 0:
		xavier_init = tf.contrib.layers.xavier_initializer(seed=seed+1)
		conv1_weights = tf.Variable(xavier_init([8,8,in_channel,24]), name='CNN_1_Weights')
		weights.append(conv1_weights)
		conv1_bias = tf.Variable(tf.constant(0.01, shape=[24]), name='CNN_1_Bias')
		weights.append(conv1_bias)
		xavier_init = tf.contrib.layers.xavier_initializer(seed=seed+2)
		conv2_weights = tf.Variable(xavier_init([4,4,24,32]), name='CNN_2_Weights')
		weights.append(conv2_weights)
		conv2_bias = tf.Variable(tf.constant(0.01, shape=[32]), name='CNN_2_Bias')
		weights.append(conv2_bias)
		xavier_init = tf.contrib.layers.xavier_initializer(seed=seed+3)
		conv3_weights = tf.Variable(xavier_init([3,3,32,32]), name='CNN_3_Weights')
		weights.append(conv3_weights)
		conv3_bias = tf.Variable(tf.constant(0.01, shape=[32]), name='CNN_3_Bias')
		weights.append(conv3_bias)

	
	conv1 = afn(tf.nn.conv2d(input=img,filter=weights[0],strides=[1,4,4,1],padding='VALID',name='cnn1')+weights[1])
	conv2 = afn(tf.nn.conv2d(input=conv1,filter=weights[2],strides=[1,2,2,1],padding='VALID',name='cnn2')+weights[3])
	# shape: 8,8,32
	conv3 = afn(tf.nn.conv2d(input=conv2,filter=weights[4],strides=[1,1,1,1],padding='VALID',name='cnn3')+weights[5])
	# shape: 7,7,32

	#batch, in_height, in_width, in_channels
	global conv3_shape_1 
	global conv3_shape_2 
	global conv3_shape_3 
	conv3_shape_1 = conv3.shape[1].value
	conv3_shape_2 = conv3.shape[2].value
	conv3_shape_3 = conv3.shape[3].value
	
	global conv2_shape_1 
	global conv2_shape_2 
	global conv2_shape_3 
	conv2_shape_1 = conv2.shape[1].value
	conv2_shape_2 = conv2.shape[2].value
	conv2_shape_3 = conv2.shape[3].value
	
	global conv1_shape_1 
	global conv1_shape_2 
	global conv1_shape_3 
	conv1_shape_1 = conv1.shape[1].value
	conv1_shape_2 = conv1.shape[2].value
	conv1_shape_3 = conv1.shape[3].value


	#conv4 = slim.conv2d(inputs=conv3,num_outputs=out_channel,activation_fn=tf.nn.sigmoid,
	#					kernel_size=[7,7],stride=[1,1],
	#					padding='VALID',scope=("CNN_4"))

	return tf.contrib.layers.flatten(conv3), weights
	#return conv3




def Deconv(input, height, width,network_name,in_channel=None,out_channel = None,afn=tf.nn.elu):

	img = tf.reshape(input,shape=[-1,conv3_shape_1,conv3_shape_2,conv3_shape_3]) 
	# This depends on the cnn output before flatten
	batch_size = tf.shape(img)[0]
	scope = network_name + '_Deconv'

	xavier_init = tf.contrib.layers.xavier_initializer(seed=seed+4)
	dconv1_weights = tf.Variable(xavier_init([3,3,32,32]), name='DCNN_1_Weights')
	dconv1_bias = tf.Variable(tf.constant(0.01, shape=[32]), name='DCNN_1_Bias')
	dconv1 = afn(tf.nn.conv2d_transpose(img,filter=dconv1_weights,output_shape=[batch_size,conv2_shape_1,conv2_shape_2,conv2_shape_3],
									strides=[1,1,1,1],padding='VALID',name='dcnn1')+dconv1_bias)
	
	xavier_init = tf.contrib.layers.xavier_initializer(seed=seed+5)
	dconv2_weights = tf.Variable(xavier_init([4,4,24,32]), name='DCNN_2_Weights')
	dconv2_bias = tf.Variable(tf.constant(0.01, shape=[24]), name='DCNN_2_Bias')
	dconv2 = afn(tf.nn.conv2d_transpose(dconv1,filter=dconv2_weights,output_shape=[batch_size,conv1_shape_1,conv1_shape_2,conv1_shape_3],
									strides=[1,2,2,1],padding='VALID',name='dcnn2')+dconv2_bias)
	
	xavier_init = tf.contrib.layers.xavier_initializer(seed=seed+6)
	dconv3_weights = tf.Variable(xavier_init([8,8,out_channel,24]), name='DCNN_3_Weights')
	dconv3_bias = tf.Variable(tf.constant(0.01, shape=[3]), name='CNN_1_Bias')
	dconv3 = tf.nn.sigmoid(tf.nn.conv2d_transpose(dconv2,filter=dconv3_weights,output_shape=[batch_size,height,width,out_channel],
									strides=[1,4,4,1],padding='VALID',name='dcnn3')+dconv3_bias)
	


	return dconv3, [dconv1_weights,dconv1_bias,dconv2_weights,dconv2_bias,dconv3_weights,dconv3_bias]


def Predictor(state, action,action_space, height, width, out_channel, network_name, state_size=None):
	scope = network_name+'_Predictor'
	#state_size = tf.shape(state)[1]
	xavier_init = tf.contrib.layers.xavier_initializer(seed=seed+7)
	in_state_weights = tf.Variable(xavier_init([state_size,state_size]), name=(scope+'_state_W'))
	in_state_bias = tf.Variable(tf.constant(0.01, shape=[state_size]), name=(scope+'_state_b'))
	
	xavier_init = tf.contrib.layers.xavier_initializer(seed=seed+8)
	in_action_weights = tf.Variable(xavier_init([action_space,state_size]), name=(scope+'_action_W'))
	in_action_bias = tf.Variable(tf.constant(0.01, shape=[state_size]), name=(scope+'_action_b'))
	
	deconv_input_state = tf.matmul(state,in_state_weights) + in_state_bias
	deconv_input_action = tf.matmul(action,in_action_weights) + in_action_bias

	pred_w = [in_state_weights,in_state_bias,in_action_weights,in_action_bias]

	deconv_input = deconv_input_state * deconv_input_action
	deconv_output, dconv_w = Deconv(deconv_input,height,width,network_name, out_channel=out_channel)
	pred_w += dconv_w

	return deconv_output, pred_w









