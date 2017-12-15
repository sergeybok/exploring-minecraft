
import tensorflow as tf 
from tensorflow.contrib import slim
import numpy as np 





def CNN(input, height, width,in_channel, out_channel,network_name,afn=tf.nn.elu):

	img = tf.reshape(input, shape=[-1, height, width, in_channel])

	conv1 = slim.conv2d(inputs=img,num_outputs=32,activation_fn=afn,
						kernel_size=[8,8],stride=[4,4],
						padding='VALID',scope=(network_name+"_CNN_1"))

	conv2 = slim.conv2d(inputs=conv1,num_outputs=64,activation_fn=afn,
						kernel_size=[4,4],stride=[2,2],
						padding='VALID',scope=(network_name+"_CNN_2"))

	conv3 = slim.conv2d(inputs=conv2,num_outputs=out_channel,activation_fn=afn,
						kernel_size=[5,5],stride=[1,1],
						padding='VALID',scope=(network_name+"_CNN_3"))

	#conv4 = slim.conv2d(inputs=conv3,num_outputs=out_channel,activation_fn=tf.nn.sigmoid,
	#					kernel_size=[7,7],stride=[1,1],
	#					padding='VALID',scope=(network_name+"_CNN_4"))

	return tf.contrib.layers.flatten(conv3)





def Deconv(input, height, width,network_name,in_channel=64,out_channel=3):

	img = tf.reshape(input,shape=[-1,5,5,in_channel]) # This depends on the cnn output before flatten

	#

	scope = network_name + '_Deconv'

	dconv1 = slim.conv2d_transpose(inputs=img,num_outputs=64,stride=[1,1],
							kernel_size=[5,5],activation_fn=tf.nn.elu,
							padding='VALID',reuse=True,scope=(scope+'_1'))
	dconv2 = slim.conv2d_transpose(inputs=dconv1,num_outputs=32,stride=[2,2],
							kernel_size=[4,4],activation_fn=tf.nn.elu,
							padding='VALID',reuse=True,scope=(scope+'_2'))
	dconv3 = slim.conv2d_transpose(inputs=dconv2,num_outputs=3,stride=[4,4],
							kernel_size=[8,8],activation_fn=tf.nn.elu,
							padding='VALID',reuse=True,scope=(scope+'_3'))
	#dconv4 = slim.conv2d_transpose(inputs=dconv3,num_outputs=out_channel,
	#						stride=[4,4],kernel_size=[8,8],activation_fn=tf.nn.sigmoid,
	#						padding='VALID',reuse=True,scope=(scope+'_4'))

	return dconv3




def Predictor(state, action, height, width, network_name, state_size=1600): 
	scope = network_name+'_Predictor'
	#state_size = tf.shape(state)[1]

	deconv_input_state = slim.fully_connected(
						inputs=state,
						num_outputs=state_size,
						activation_fn=tf.nn.sigmoid,
						reuse=True,
						scope=(scope+'_state')) 
	deconv_input_action1 = slim.fully_connected(
							inputs=action,
							num_outputs=state_size,
							activation_fn=tf.nn.tanh,
							reuse=True,
							scope=(scope+'_action1'))

	deconv_input = deconv_input_state * deconv_input_action1
	
	return Deconv(deconv_input,height,width,network_name)









