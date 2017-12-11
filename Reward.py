import tensorflow as tf
import numpy as np

import Perception 




def Evaluate_Predictor(states, actions,state_size, window_size,ht,wd,out_channel,network_name):
	# evaluates the accuracy of the predictor on cached history
	predictions = [None]*window_size
	for i in range(window_size):
		predictions[i] = Perception.Predictor(states[i],state_size,actions[i],
									ht,wd,out_channel,network_name)
	
	return predictions




def Get_Intrinsic_Reward(predictions_t, predictions_tm1,frames,window_size):

	reward = tf.reduce_mean(tf.square(predictions_tm1[0]-frames[0])) - tf.reduce_mean(tf.square(predictions_t[0]-frames[0]))

	for i in range(1,window_size):
		reward += tf.reduce_mean(tf.square(predictions_tm1[i]-frames[i])) - tf.reduce_mean(tf.square(predictions_t[i]-frames[i]))


	return reward 





























