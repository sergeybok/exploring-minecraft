import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import pickle
import time
import datetime
import sys

import Perception
import Reward

# from environments import maze_environment
import maze_environment


use_intrinsic_reward = True
use_complete_random_agent = False
historical_sample_size = 100


class Qnetwork():
    def __init__(self, frame_height=None, frame_width=None, frame_channels=None, total_num_actions=None):
        flattened_frame_size = frame_height*frame_width*frame_channels
        self.flattened_image = tf.placeholder(shape=[None, flattened_frame_size], dtype=tf.float32)
        #[batch, in_height, in_width, in_channels]
        #[filter_height, filter_width, in_channels, out_channels]
        # Must have strides[0] = strides[3] = 1. For the most common case of the same horizontal and vertices strides, strides = [1, stride, stride, 1].

        self.state_feature_vector, self.CNN_params = Perception.CNN(input=self.flattened_image,height=frame_height,width=frame_width,in_channel=frame_channels,out_channel=32,weights=[])

        #NOTE :::: Split is not really required, also even if you use split, it should be done on the dimension of feature maps. Also the weight matrices have to be correctly shaped.
        with tf.variable_scope("advantage_stream"):
            self.streamAC = self.state_feature_vector
            self.streamA = tf.contrib.layers.flatten(self.streamAC)
            xavier_init = tf.contrib.layers.xavier_initializer(seed=8739)
            self.AW1 = tf.Variable(xavier_init([self.streamA.shape[1].value, 512]), name='FC_1_Weights')
            self.ABias1 = tf.Variable(tf.constant(0.1, shape=[512]), name='FC_1_Bias')
            self.Advantage_FC1 = tf.matmul(self.streamA, self.AW1) + self.ABias1
            self.Advantage_FC1 = tf.nn.relu(self.Advantage_FC1)
            xavier_init = tf.contrib.layers.xavier_initializer(seed=8536)
            self.AW2 = tf.Variable(xavier_init([self.Advantage_FC1.shape[1].value, total_num_actions]), name='FC_2_Weights')
            self.ABias2 = tf.Variable(tf.constant(0.1, shape=[total_num_actions]), name='FC_2_Bias')
            self.Advantage = tf.matmul(self.Advantage_FC1, self.AW2) + self.ABias2
            self.advantage_w = [self.AW1, self.ABias1,self.AW2,self.ABias2]
        with tf.variable_scope("value_stream"):
            self.streamVC = self.state_feature_vector
            self.streamV = tf.contrib.layers.flatten(self.streamVC)
            xavier_init = tf.contrib.layers.xavier_initializer(seed=8635)
            self.VW1 = tf.Variable(xavier_init([self.streamV.shape[1].value, 512]), name='FC_1_Weights')
            self.VBias1 = tf.Variable(tf.constant(0.1, shape=[512]), name='FC_1_Bias')
            self.Value_FC1 = tf.matmul(self.streamV, self.VW1) + self.VBias1
            self.Value_FC1 = tf.nn.relu(self.Value_FC1)
            xavier_init = tf.contrib.layers.xavier_initializer(seed=8267)
            self.VW2 = tf.Variable(xavier_init([self.Value_FC1.shape[1].value, 1]), name='FC_2_Weights')
            self.VBias2 = tf.Variable(tf.constant(0.1, shape=[1]), name='FC_2_Bias')
            self.Value = tf.matmul(self.Value_FC1, self.VW2) + self.VBias2
            self.value_w = [self.VW1,self.VBias1,self.VW2,self.VBias2]
        # NOTE ::: Add the state value and advantage value to get the q values but note that we subtract the average advantage value from advantage value of all actions.
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
        # NOTE :: we take argmax over advantage values instead of Q values
        self.predict = tf.argmax(self.Advantage, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, total_num_actions, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        network_scope_name = tf.get_variable_scope().name
        if(network_scope_name=='Q_main'):
            Q_vars = self.advantage_w + self.value_w
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
            gvs = self.optimizer.compute_gradients(self.loss,var_list=Q_vars)
            with tf.variable_scope("gradient_clipping"):
                capped_gvs = [(tf.clip_by_norm(grad, 10.0), var) for grad, var in gvs]
            self.train_Q_op = self.optimizer.apply_gradients(capped_gvs)
            gvs_cnn = self.optimizer.compute_gradients(self.loss,var_list=self.CNN_params)
            with tf.variable_scope("gradient_clipping_cnn"):
                capped_gvs = [(tf.clip_by_norm(grad, 10.0), var) for grad, var in gvs_cnn]
            self.train_cnn_op = self.optimizer.apply_gradients(capped_gvs)
            # self.train_op = self.optimizer.minimize(self.loss)




# ### Experience Replay
# This class allows us to store experiences and sample then randomly to train the network.
class experience_buffer():
    def __init__(self, buffer_size=2000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        #Since we have 5 elements in the experience : s, a, r, s1, is_terminal_flag, therefore we reshape it as [size, 5]
        if(size>len(self.buffer)):
            size = len(self.buffer)
        return (np.reshape(np.array(random.sample(self.buffer, size)), [size, 5]), size)

    def is_empty(self):
        return len(self.buffer) == 0


def target_network_update_op(Q_main_variables, Q_target_variables, tau):
    target_network_update_ops = []
    with tf.variable_scope("target_network_update_ops"):
        for main_network_var, target_network_var in zip(sorted(Q_main_variables, key=lambda v: v.name), sorted(Q_target_variables, key=lambda v: v.name)):
            #print('main var name {0}'.format(main_network_var.name))
            #print('tgt var name {0}'.format(target_network_var))
            assign_value = (main_network_var.value()*tau) + ((1 - tau)*target_network_var.value())
            update_op = target_network_var.assign(assign_value)
            target_network_update_ops.append(update_op)
        grouped_target_network_update_op = tf.group(*target_network_update_ops)
    return grouped_target_network_update_op

# ### Training the network
# Setting all the training parameters
batch_size = 100  # How many experiences to use for each training step.
gamma_discount_factor = .9999  # Discount factor on the target Q-values
startE = 0.5  # Starting chance of random action
endE = 0.05  # Final chance of random action
annealing_steps = 7000  # How many steps of training to reduce startE to endE.
batch_size_deconv_compressor = 10
intrinsic_reward_rescaling_factor = 2
num_episodes = 5  # How many episodes of game environment to train network with.
if(use_complete_random_agent):
    update_freq_per_episodes = num_episodes # How often to perform a training step.
else:
    update_freq_per_episodes = 1  # How often to perform a training step.
pre_train_steps = 100  # How many steps of random actions before training begins.
max_actions_per_episode = 160  # The max allowed length of our episode.
load_model = False  # Whether to load a saved model.
path_QNetwork = "./curiosity_model/dqn_model"  # The path to save our model to.
path_Frame_Predictor = "./curiosity_model/frame_predictor_model"  # The path to save our model to.
model_saving_freq = 50
# h_size = 512  # The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001  # Rate to update target network toward primary network
num_previous_frames = 4
previous_frames = []


QNetwork_graph = tf.Graph()
#Frame_Predictor_graph = tf.Graph()

port_number = 10000
maze_env = maze_environment.environment(port_number)
frame_height = maze_env.video_height
frame_width = maze_env.video_width
frame_channels = maze_env.video_channels
#TODO define whether the actions are continous or discrete




total_num_actions = maze_env.total_num_actions

with tf.variable_scope("Q_main") as Q_main_scope:
    mainQN = Qnetwork(frame_height=frame_height, frame_width=frame_width, frame_channels=frame_channels, total_num_actions=total_num_actions)
with tf.variable_scope("Q_target") as Q_target_scope:
    targetQN = Qnetwork(frame_height=frame_height, frame_width=frame_width, frame_channels=frame_channels, total_num_actions=total_num_actions)
   

saver_QNetwork = tf.train.Saver()

#IMPORTANT NOTE::: When a scope is passed to the following tf.get_collection it returns only those trainable variables which are named.
# So it is important to name all the trainable variables.
Q_main_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=Q_main_scope.name)
Q_target_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=Q_target_scope.name)
grouped_target_network_update_op = target_network_update_op(Q_main_variables, Q_target_variables, tau)


if use_intrinsic_reward:
    curiosity = Reward.Compressor(frame_height=frame_height, frame_width=frame_width, frame_channels=frame_channels, 
                            state_feature_size=mainQN.state_feature_vector.shape[1].value, CNN_W=mainQN.CNN_params,
                            total_num_actions=total_num_actions, network_name='compressor')
    

myBuffer = experience_buffer()

# Set the rate of random action decrease.
e = startE
stepDrop = float(startE - endE)/annealing_steps

# create lists to contain total rewards and steps per episode
steps_taken_per_episode_list = []
reward_per_episode_list = []
mean_reward_window_len = 10
mean_reward_per_episode_window_list = []
total_steps = 0

# Make a path for our model to be saved in.
if not os.path.exists(path_QNetwork):
    os.makedirs(path_QNetwork)


curr_episode_total_reward_placeholder = tf.placeholder(tf.float32, name='curr_episode_total_reward')
curr_episode_reward_summary = tf.summary.scalar("Per episode reward", curr_episode_total_reward_placeholder)
mean_reward_over_window_placeholder = tf.placeholder(tf.float32, name='mean_reward_over_window')
mean_reward_over_window_summary = tf.summary.scalar("Mean episodic reward over window", mean_reward_over_window_placeholder)


init_QNetwork_graph = tf.global_variables_initializer()
sess_QNetwork = tf.Session()


tf_graph_file_name = './curiosity_model/tf_graphs/subgoal_{0}_goal_{1}_size_{2}'.format(maze_env.reward_subgoal, maze_env.reward_goal, maze_env.maze_size)
if not os.path.exists(tf_graph_file_name):
    os.makedirs(tf_graph_file_name)
writer_op_QNetwork = tf.summary.FileWriter(tf_graph_file_name, sess_QNetwork.graph)
sess_QNetwork.run(init_QNetwork_graph)

curr_episode_total_reward_summary = tf.Summary()
if load_model == True:
    print('Loading Model...')
    ckpt_QNetwork = tf.train.get_checkpoint_state(path_QNetwork)
    saver.restore(sess_QNetwork, ckpt_QNetwork.model_checkpoint_path)
    

start_time = time.time()
for episode_num in range(num_episodes):
    curr_episode_total_reward = 0
    maze_env.get_maze()
    episodeBuffer = experience_buffer()
    s = maze_env.get_current_state()
    is_terminal_flag = False
    steps_taken_per_episode = 0
    # The Q-Network
    #NOTE ::: We can condition the below while loop on either a pre-defined number of maximum action or wait for the environment episode to get over when the agent runs out of the mission time.
    #NOTE ::: I have conditioned the while loop on the mission time.
    # while steps_taken_per_episode < max_epLength:  # If the agent takes longer than 50 moves to reach the end of the maze, end the trial.
    while(maze_env.world_state.is_mission_running and steps_taken_per_episode<max_actions_per_episode):
        steps_taken_per_episode += 1
        if(not use_intrinsic_reward):
            cnn_features_state_s = None
            if(np.random.rand(1) < e or total_steps < pre_train_steps):
                # Choose an action by greedily (with e chance of random action) from the Q-network
                a = np.random.randint(0, total_num_actions)
            else:
                a = sess_QNetwork.run(mainQN.predict, feed_dict={mainQN.flattened_image: [s]})[0]
        else:
            a, cnn_features_state_s = sess_QNetwork.run([mainQN.predict, mainQN.state_feature_vector], feed_dict={mainQN.flattened_image: [s]})
            a = a[0]
        if(use_complete_random_agent):
            a = np.random.randint(0, total_num_actions)
        action_result = maze_env.take_action(a)
        if(action_result):
            is_terminal_flag = action_result[2]
            if(not(is_terminal_flag)):
                s1 = action_result[0]
                # TODO save cnn_features_state_s1, that is the convolved output of next frame used for predictor loss
                r = action_result[1]
            else:
                #TODO ::: Update this else condition when I can get the terminal state frame and terminal state rewards
                break
        else:
            break
        total_steps += 1

        episodeBuffer.add(np.reshape(np.array([s, a, r, s1, is_terminal_flag]), [1, 5]))  # Save the experience to our episode buffer.
        # Since we have 6 elements in the experience : s, a, r, s1, is_terminal_flag, cnn_s, therefore we reshape it as [size, 6]

        curr_episode_total_reward += r
        s = s1

        # if(False):
        if total_steps > pre_train_steps:
            # NOTE::: We are reducing the epsilon of exploration after every action we take, not after every episode, so the epsilon decreases within 1 episode
            if e > endE:
                e -= stepDrop

        if is_terminal_flag == True:
            break

    if use_intrinsic_reward and not myBuffer.is_empty():
        number_batches = len(episodeBuffer.buffer)//batch_size_deconv_compressor
        for i in range(number_batches):
            curr_batch = episodeBuffer.buffer[i*batch_size_deconv_compressor:(i+1)*batch_size_deconv_compressor]
            curr_batch = np.reshape(np.array(curr_batch), [len(curr_batch), 5])
            curr_batch_state_features = np.vstack(curr_batch[:, 0])
            curr_batch_actions = curr_batch[:, 1]
            curr_batch_states_tp1 = np.vstack(curr_batch[:, 3])

            sample_, _ = myBuffer.sample(historical_sample_size)
            state_feature_sample = np.vstack(sample_[:,0])
            # action_sample = np.vstack(sample_[:,1]).astype(np.uint8)
            action_sample = sample_[:,1]
            state_tp1 = np.vstack(sample_[:,3])

            pred_tm1 = curiosity.predict_next_state(sess_QNetwork, state_feature_sample, action_sample)

            l = curiosity.train(sess_QNetwork,curr_batch_state_features, curr_batch_actions, curr_batch_states_tp1)

            pred_t = curiosity.predict_next_state(sess_QNetwork,state_feature_sample,action_sample)
            intrinsic_r = curiosity.get_reward(predictions_t=pred_t,predictions_tm1=pred_tm1,targets=state_tp1)
            intrinsic_r = min(intrinsic_r * intrinsic_reward_rescaling_factor, 1)

            curr_batch = episodeBuffer.buffer[i * batch_size_deconv_compressor:(i + 1) * batch_size_deconv_compressor]
            for list_index in range(len(curr_batch)):
                curr_batch[list_index][2] += intrinsic_r

    #TODO Figure out why is it not being trained for the episode 1
    if total_steps > pre_train_steps:
        if (episode_num % (update_freq_per_episodes) == 0 and episode_num > 0):
            for batch_num in range(10):
                trainBatch, actual_sampled_size = myBuffer.sample(batch_size)  # Get a random batch of experiences.
                # Below we perform the Double-DQN update to the target Q-values
                Q1 = sess_QNetwork.run(mainQN.predict, feed_dict={mainQN.flattened_image: np.vstack(trainBatch[:, 3])})
                Q2 = sess_QNetwork.run(targetQN.Qout, feed_dict={targetQN.flattened_image: np.vstack(trainBatch[:, 3])})
                # NOTE ::: the use of end_multiplier --- the is_terminal_flag gets stored as 1 or 0(True or False),
                # NOTE ::: Done if the is_terminal_flag is true i.e 1, we define the end_multiplier as 0, if the is_terminal_flag is false i.e 0, we define the end_multiplier as 1
                end_multiplier = -(trainBatch[:, 4] - 1)
                doubleQ = Q2[range(actual_sampled_size), Q1]
                targetQ = trainBatch[:, 2] + (gamma_discount_factor * doubleQ * end_multiplier)
                # Update the network with our target values.
                # NOTE ::: it is important to recalculate the Q values of the states in the experience replay and then get the gradient w.r.t difference b/w recalculated values and targets
                # NOTE ::: otherwise it defeats the purpose of experience replay, also we are not storing the Q values for this reason
                _,_ = sess_QNetwork.run([mainQN.train_Q_op,mainQN.train_cnn_op], feed_dict={mainQN.flattened_image: np.vstack(trainBatch[:, 0]), mainQN.targetQ: targetQ, mainQN.actions: trainBatch[:, 1]})
                _ = sess_QNetwork.run(grouped_target_network_update_op)

    print('Episode : '+str(episode_num)+' Total reward : '+str(curr_episode_total_reward)+' Total steps : '+str(steps_taken_per_episode))
    myBuffer.add(episodeBuffer.buffer)
    summary_val, = sess_QNetwork.run([curr_episode_reward_summary], feed_dict={curr_episode_total_reward_placeholder: curr_episode_total_reward})
    writer_op_QNetwork.add_summary(summary_val, episode_num + 1)
    reward_per_episode_list.append(curr_episode_total_reward)
    mean_reward_over_window = sum(reward_per_episode_list[-mean_reward_window_len:]) / min(len(reward_per_episode_list), mean_reward_window_len)
    summary_val, = sess_QNetwork.run([mean_reward_over_window_summary], feed_dict={mean_reward_over_window_placeholder: mean_reward_over_window})
    writer_op_QNetwork.add_summary(summary_val, episode_num + 1)
    mean_reward_per_episode_window_list.append(mean_reward_over_window)
    steps_taken_per_episode_list.append(steps_taken_per_episode)
    # Periodically save the model.
    if(episode_num % model_saving_freq == 0 and episode_num>0):
        saver_QNetwork.save(sess_QNetwork, path_QNetwork + '/model-' + str(episode_num) + '.ckpt')
        print("Saved Model after episode : "+str(episode_num))
    if len(reward_per_episode_list) % 10 == 0:
        print('Total steps taken till now, mean reward per episode, current epsilon :::::: ')
        print(str(total_steps)+', '+str(np.mean(reward_per_episode_list))+', '+str(e))

end_episode_time = time.time()
duration = end_episode_time-start_time
duration = datetime.timedelta(seconds=duration)
print('Total running time is {0}'.format(duration))
saver_QNetwork.save(sess_QNetwork, path_QNetwork + '/model-' + str(episode_num) + '.ckpt')
writer_op_QNetwork.close()

print("Percent of succesful episodes: " + str(sum(reward_per_episode_list) / num_episodes) + "%")

# ### Checking network learning
# Mean reward over time

reward_per_episode_list = np.array(reward_per_episode_list)
rMean = np.average(reward_per_episode_list)
print('Mean reward is '+str(rMean))

results_file_name = './curiosity_model/DQN_results.pickle'
fp = open(results_file_name, 'w')
results_dict = {'reward_per_episode_list':reward_per_episode_list, 'mean_reward_per_episode_window_list':mean_reward_per_episode_window_list, 'steps_taken_per_episode_list':steps_taken_per_episode_list}
pickle.dump(results_dict, fp)
fp.close()

plt.plot(reward_per_episode_list)

