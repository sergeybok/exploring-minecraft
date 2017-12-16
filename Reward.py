import tensorflow as tf
import numpy as np

import Perception 




class Compressor:

	def __init__(self,frame_height=None, frame_width=None, frame_channels=None,state_feature_size=1600, total_num_actions=None, network_name=None):

		self.state_feature = tf.placeholder(tf.float32,shape=[None,state_feature_size],name='Compressor_state_input')
		self.action = tf.placeholder(tf.uint8,shape=[None],name='Compressor_action_input')
		self.action_one_hot = tf.one_hot(self.action, total_num_actions, dtype=tf.float32)
		self.state_tp1 = tf.placeholder(tf.float32,shape=[None,frame_height*frame_width*frame_channels],name='Compressor_state_tp1_input')

		self.predicted_image = Perception.Predictor(state=self.state_feature,
												state_size=state_feature_size,
												action=self.action_one_hot,
												height=frame_height,
												width=frame_width,
												network_name='Compressor')

		self.prediction_flattened = tf.contrib.layers.flatten(self.predicted_image)

		self.predictor_loss = tf.reduce_mean(tf.square(self.state_tp1 - self.prediction_flattened))
		self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
		gvs = self.optimizer.compute_gradients(self.predictor_loss)
		#if gvs == None:
		#	print ('helll oooo ooo ======')
		# capped_gvs = [(tf.clip_by_norm(grad, 10.0), var) for grad, var in gvs]
		capped_gvs = gvs
		self.train_pred = self.optimizer.apply_gradients(capped_gvs)


	def train(self,sess,states, actions, states_tp1):
		_, loss = sess.run([self.train_pred, self.predictor_loss], feed_dict={self.state_feature:states,self.action:actions,self.state_tp1:states_tp1})
		return loss


	def predict_next_state(self,sess,states,actions):
		prediction_flattened_value, = sess.run([self.prediction_flattened],
			feed_dict={self.state_feature:states,self.action:actions})
		return prediction_flattened_value


	def get_reward(self,predictions_t,predictions_tm1,targets):
		loss_tm1 = np.mean(np.square(predictions_tm1 - targets))
		loss_t = np.mean(np.square(predictions_t - targets))
		improvement = (loss_tm1 - loss_t)
		if improvement < 0 :
			return 0
		return improvement






























