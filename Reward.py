import tensorflow as tf
import numpy as np

import Perception 




class Compressor:

	def __init__(self,frame_height=None, frame_width=None, frame_channels=None,state_size=3136, total_num_actions=None, network_name=None):

		self.state = tf.placeholder(tf.float32,shape=[None,state_size])
		self.action = tf.placeholder(tf.float32,shape=[None,total_num_actions])
		self.state_tp1 = tf.placeholder(tf.float32,shape=[None,frame_height,frame_width,frame_channels])

		self.prediction = Perception.Predictor(state=self.state,
												state_size=state_size,
												action=self.action,
												height=frame_height,
												width=frame_width,
												network_name='Compressor')

		self.predictor_loss = tf.reduce_mean(tf.square(self.state_tp1 - self.prediction))
		self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
		gvs = self.optimizer.compute_gradients(self.predictor_loss)
		#if gvs == None:
		#	print ('helll oooo ooo ======')
		#capped_gvs = [(tf.clip_by_norm(grad, 10.0), var) for grad, var in gvs]
		capped_gvs = gvs
		self.train_pred = self.optimizer.apply_gradients(capped_gvs)


	def train(self,sess,states, actions, states_tp1):
		_, loss = sess.run([self.train_pred, self.predictor_loss],{self.state:states,self.action:actions,self.state_tp1:states_tp1})
		return loss


	def predict_next_state(self,sess,states,actions):
		return sess.run([self.prediction],{self.state:states,self.action:actions})


	def get_reward(self,predictions_t,predictions_tm1,targets):
		loss_tm1 = np.mean(np.square(predictions_tm1 - targets))
		loss_t = np.mean(np.square(predictions_t - targets))
		improvement = (loss_tm1 - loss_t)
		if improvement < 0 :
			return 0
		return improvement






























