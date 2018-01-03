import tensorflow as tf
import numpy as np

import Perception 




class Compressor:

	def __init__(self,frame_height=None, frame_width=None, frame_channels=None,state_feature_size=None, total_num_actions=None, CNN_W=[], network_name=None):

		self.CNN_w = CNN_W
		self.flat_image = tf.placeholder(tf.float32,shape=[None,frame_height*frame_width*frame_channels],
											name='Compressor_state_input')
		#self.image = tf.reshape(self.image,[-1,frame_height,frame_width,frame_channels])
		#self.state_feature = tf.placeholder(tf.float32,shape=[None,state_feature_size],
		#									name='Compressor_state_input')
		self.action = tf.placeholder(tf.uint8,shape=[None],name='Compressor_action_input')
		self.action_one_hot = tf.one_hot(self.action, total_num_actions, dtype=tf.float32)
		self.state_tp1 = tf.placeholder(tf.float32,shape=[None,frame_height*frame_width*frame_channels],
										name='Compressor_state_tp1_input')

		# self.state_feature, self.CNN_w = Perception.CNN(input=self.flat_image,height=frame_height,width=frame_width,in_channel=frame_channels,out_channel=32,weights=CNN_W)
		self.state_feature, _ = Perception.CNN(input=self.flat_image,height=frame_height,width=frame_width,in_channel=frame_channels,out_channel=32,weights=self.CNN_w)

		self.predicted_image, self.compressor_weights = Perception.Predictor(state=self.state_feature,
												state_size=state_feature_size,
												action=self.action_one_hot,
												action_space=total_num_actions,
												height=frame_height,
												width=frame_width,
												out_channel=frame_channels,
												network_name='Compressor')

		self.prediction_flattened = tf.contrib.layers.flatten(self.predicted_image)

		self.predictor_loss = tf.reduce_mean(tf.square(self.state_tp1 - self.prediction_flattened))
		self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0004)
		gvs_dcnn = self.optimizer.compute_gradients(self.predictor_loss,var_list=self.compressor_weights)
		#if gvs == None:
		#	print ('helll oooo ooo ======')
		# capped_gvs = [(tf.clip_by_norm(grad, 10.0), var) for grad, var in gvs]
		capped_gvs_dcnn = gvs_dcnn
		self.train_pred = self.optimizer.apply_gradients(capped_gvs_dcnn)

		gvs_cnn = self.optimizer.compute_gradients(self.predictor_loss,var_list=self.CNN_w)
		self.train_cnn = self.optimizer.apply_gradients(gvs_cnn)


	def trainDCNN(self,sess,states, actions, states_tp1):
		_, loss = sess.run([self.train_pred, self.predictor_loss], 
						feed_dict={self.flat_image:states,self.action:actions,
								self.state_tp1:states_tp1})
		return loss

	def train(self,sess,states, actions, states_tp1):
		_, _, loss = sess.run([self.train_pred, self.train_cnn, self.predictor_loss], 
						feed_dict={self.flat_image:states,self.action:actions,
							self.state_tp1:states_tp1})
		return loss


	def predict_next_state(self,sess,states,actions):
		prediction_flattened_value, = sess.run([self.prediction_flattened],
			feed_dict={self.flat_image:states,self.action:actions})
		return prediction_flattened_value


	def get_reward(self,predictions_t,predictions_tm1,targets):
		loss_tm1 = np.mean(np.square(predictions_tm1 - targets))
		loss_t = np.mean(np.square(predictions_t - targets))
		improvement = (loss_tm1 - loss_t)
		if improvement < 0 :
			return 0
		return improvement






























