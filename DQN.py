import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import pickle

import Perception
import Reward

# from environments import maze_environment
import maze_environment


use_intrinsic_reward = True
historical_sample_size = 100


class Qnetwork():
    def __init__(self, frame_height=None, frame_width=None, frame_channels=None, total_num_actions=None, network_name=None):
        flattened_frame_size = frame_height*frame_width*frame_channels
        self.flattened_image = tf.placeholder(shape=[None, flattened_frame_size], dtype=tf.float32)
        #[batch, in_height, in_width, in_channels]
        #[filter_height, filter_width, in_channels, out_channels]
        # Must have strides[0] = strides[3] = 1. For the most common case of the same horizontal and vertices strides, strides = [1, stride, stride, 1].
        #self.reshaped_image = tf.reshape(self.flattened_image, shape=[-1, frame_height, frame_width, frame_channels])
        #self.conv1 = self.conv_layer(input_volume=self.reshaped_image, num_output_channel=32, filter_shape=[8, 8], strides_shape=[1, 4, 4, 1], padding_type='VALID', network_name=network_name, layer_name='1')
        #self.conv2 = self.conv_layer(input_volume=self.conv1, num_output_channel=64, filter_shape=[4, 4], strides_shape=[1, 2, 2, 1], padding_type='VALID', network_name=network_name, layer_name='2')
        #self.conv3 = self.conv_layer(input_volume=self.conv2, num_output_channel=64, filter_shape=[3, 3], strides_shape=[1, 1, 1, 1], padding_type='VALID', network_name=network_name, layer_name='3')
        # self.conv4 = self.conv_layer(input_volume=self.conv3, num_output_channel=64, filter_shape=[3, 3], strides_shape=[1, 1, 1, 1], padding_type='VALID', network_name=network_name, layer_name='4')

        self.state_vector = Perception.CNN(input=self.flattened_image,height=frame_height,width=frame_width,
                                            in_channel=3,out_channel=64,network_name=network_name)


        #NOTE :::: Split is not really required, also even if you use split, it should be done on the dimension of feature maps. Also the weight matrices have to be correctly shaped.
        self.streamAC = self.state_vector
        self.streamVC = self.state_vector
        self.streamA = tf.contrib.layers.flatten(self.streamAC)
        self.streamV = tf.contrib.layers.flatten(self.streamVC)
        xavier_init = tf.contrib.layers.xavier_initializer()
        # print(tf.shape(self.streamA)[1])
        self.AW1 = tf.Variable(xavier_init([self.streamA.shape[1].value, 512]))
        self.ABias1 = tf.Variable(tf.constant(0.1, shape=[512]))
        self.VW1 = tf.Variable(xavier_init([self.streamV.shape[1].value, 512]))
        self.VBias1 = tf.Variable(tf.constant(0.1, shape=[512]))
        self.Advantage_FC1 = tf.matmul(self.streamA, self.AW1)+self.ABias1
        self.Advantage_FC1 = tf.nn.relu(self.Advantage_FC1)
        self.Value_FC1 = tf.matmul(self.streamV, self.VW1)+self.VBias1
        self.Value_FC1 = tf.nn.relu(self.Value_FC1)
        self.AW2 = tf.Variable(xavier_init([self.Advantage_FC1.shape[1].value, total_num_actions]))
        self.ABias2 = tf.Variable(tf.constant(0.1, shape=[total_num_actions]))
        self.VW2 = tf.Variable(xavier_init([self.Value_FC1.shape[1].value, 1]))
        self.VBias2 = tf.Variable(tf.constant(0.1, shape=[1]))
        self.Advantage = tf.matmul(self.Advantage_FC1, self.AW2)+self.ABias2
        self.Value = tf.matmul(self.Value_FC1, self.VW2)+self.VBias2

        # NOTE ::: Add the state value and advantage value to get the q values but note that we subtract the average advantage value from advantage value of all actions.
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
        # We predict the action using argmax of advantages and not on Q values, anyway the argmax will be the same from both advantage and q values
        # But taking the argmax from the advantage values saves us the computation of V values
        # self.predict = tf.argmax(self.Qout, 1)
        self.predict = tf.argmax(self.Advantage, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, total_num_actions, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        if(network_name=='main'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
            gvs = self.optimizer.compute_gradients(self.loss)
            capped_gvs = [(tf.clip_by_norm(grad, 10.0), var) for grad, var in gvs]
            self.train_op = self.optimizer.apply_gradients(capped_gvs)
            # self.train_op = self.optimizer.minimize(self.loss)



# ### Experience Replay
# This class allows us to store experiences and sample then randomly to train the network.
class experience_buffer():
    def __init__(self, buffer_size=1000):
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


# These functions allow us to update the parameters of our target network with those of the primary network.
# tfVars contains the variables of the primary network and of the target network in the respective order.
# Therefore we update the values of the target network variables using the value of primary network variables and the target network variables itself.
def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:total_vars // 2]):
        op_holder.append(tfVars[idx + total_vars // 2].assign(
            (var.value() * tau) + ((1 - tau) * tfVars[idx + total_vars // 2].value())))
    return op_holder

def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)

# ### Training the network
# Setting all the training parameters
batch_size = 100  # How many experiences to use for each training step.
update_freq_per_episodes = 1  # How often to perform a training step.
gamma_discount_factor = .99  # Discount factor on the target Q-values
startE = 0.5  # Starting chance of random action
endE = 0.05  # Final chance of random action
annealing_steps = 5000.  # How many steps of training to reduce startE to endE.
num_episodes = 200 # How many episodes of game environment to train network with.
pre_train_steps = 100  # How many steps of random actions before training begins.
# max_epLength = 500  # The max allowed length of our episode.
load_model = False  # Whether to load a saved model.
path = "./dqn_model"  # The path to save our model to.
model_saving_freq = 10
# h_size = 512  # The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001  # Rate to update target network toward primary network

#TODO why is reset default graph required??
tf.reset_default_graph()
maze_env = maze_environment.environment()
frame_height = maze_env.video_height
frame_width = maze_env.video_width
frame_channels = maze_env.video_channels
#TODO define whether the actions are continous or discrete
total_num_actions = maze_env.total_num_actions
mainQN = Qnetwork(frame_height=frame_height, frame_width=frame_width, frame_channels=frame_channels, total_num_actions=total_num_actions, network_name='main')
targetQN = Qnetwork(frame_height=frame_height, frame_width=frame_width, frame_channels=frame_channels, total_num_actions=total_num_actions, network_name='target')

if use_intrinsic_reward:
    curiosity = Reward.Compressor(frame_height=frame_height, frame_width=frame_width, frame_channels=frame_channels, 
                                    total_num_actions=total_num_actions, network_name='compressor')

init = tf.global_variables_initializer()
saver = tf.train.Saver()
trainables = tf.trainable_variables()
targetOps = updateTargetGraph(trainables, tau)
myBuffer = experience_buffer()

# Set the rate of random action decrease.
e = startE
stepDrop = (startE - endE) / annealing_steps

# create lists to contain total rewards and steps per episode
steps_taken_per_episode_list = []
reward_per_episode_list = []
mean_reward_window_len = 10
mean_reward_per_episode_window_list = []
total_steps = 0

# Make a path for our model to be saved in.
if not os.path.exists(path):
    os.makedirs(path)

curr_episode_total_reward_placeholder = tf.placeholder(tf.float32, name='curr_episode_total_reward')
curr_episode_reward_summary = tf.summary.scalar("curr_episode_total_reward", curr_episode_total_reward_placeholder)
mean_reward_over_window_placeholder = tf.placeholder(tf.float32, name='mean_reward_over_window')
mean_reward_over_window_summary = tf.summary.scalar("mean_reward_over_window", mean_reward_over_window_placeholder)


with tf.Session() as sess:
    writer_op = tf.summary.FileWriter('./tf_graphs', sess.graph)
    sess.run(init)
    curr_episode_total_reward_summary = tf.Summary()
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)
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
        while(maze_env.world_state.is_mission_running):
            steps_taken_per_episode += 1
            # Choose an action by greedily (with e chance of random action) from the Q-network
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                a = np.random.randint(0, total_num_actions)
            else:
                a = sess.run(mainQN.predict, feed_dict={mainQN.flattened_image: [s]})[0]
            action_result = maze_env.take_action(a)
            if(action_result):
                is_terminal_flag = action_result[2]
                if(not(is_terminal_flag)):
                    s1 = action_result[0]
                    r = action_result[1]
                else:
                    #TODO ::: Update this else condition when I can get the terminal state frame and terminal state rewards
                    break
            else:
                break
            total_steps += 1
            episodeBuffer.add(np.reshape(np.array([s, a, r, s1, is_terminal_flag]), [1, 5]))  # Save the experience to our episode buffer.
            # Since we have 5 elements in the experience : s, a, r, s1, is_terminal_flag, therefore we reshape it as [size, 5]
            
            if use_intrinsic_reward:
                sample_ = episodeBuffer.sample(historical_sample_size)
                state_sample = sample_[:,0]
                action_sample = sample_[:,1]
                state_tp1 = sample_[:,3]
                pred_tm1 = curiosity.predict_next_state(sess,state_sample,action_sample)
                l = curiosity.train(sess,state_sample, action_sample, state_tp1)
                pred_t = curiosity.predict_next_state(sess,state_sample,action_sample)
                intrinsic_r = curiosity.get_reward(pred_t,pred_tm1,state_tp1)
                r += intrinsic_r

            curr_episode_total_reward += r
            s = s1

            if total_steps > pre_train_steps:
                # NOTE::: We are reducing the epsilon of exploration after every action we take, not after every episode, so the epsilon decreases within 1 episode
                if e > endE:
                    e -= stepDrop

            if is_terminal_flag == True:
                break

        if total_steps > pre_train_steps:
            if (episode_num % (update_freq_per_episodes) == 0 and episode_num > 0):
                for batch_num in range(10):
                    # print('current overall experience buffer size is '+str(len(myBuffer.buffer)))
                    # print('sample a batch size of '+str(batch_size))
                    trainBatch, actual_sampled_size = myBuffer.sample(batch_size)  # Get a random batch of experiences.
                    # Below we perform the Double-DQN update to the target Q-values
                    Q1 = sess.run(mainQN.predict, feed_dict={mainQN.flattened_image: np.vstack(trainBatch[:, 3])})
                    Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.flattened_image: np.vstack(trainBatch[:, 3])})
                    # NOTE ::: the use of end_multiplier --- the is_terminal_flag gets stored as 1 or 0(True or False),
                    # NOTE ::: Done if the is_terminal_flag is true i.e 1, we define the end_multiplier as 0, if the is_terminal_flag is false i.e 0, we define the end_multiplier as 1
                    end_multiplier = -(trainBatch[:, 4] - 1)
                    doubleQ = Q2[range(actual_sampled_size), Q1]
                    targetQ = trainBatch[:, 2] + (gamma_discount_factor * doubleQ * end_multiplier)
                    # Update the network with our target values.
                    # NOTE ::: it is important to recalculate the Q values of the states in the experience replay and then get the gradient w.r.t difference b/w recalculated values and targets
                    # NOTE ::: otherwise it defeats the purpose of experience replay, also we are not storing the Q values for this reason
                    _ = sess.run(mainQN.train_op, feed_dict={mainQN.flattened_image: np.vstack(trainBatch[:, 0]), mainQN.targetQ: targetQ, mainQN.actions: trainBatch[:, 1]})
                    updateTarget(targetOps, sess)  # Update the target network toward the primary network.

        print('Episode : '+str(episode_num)+' Total reward : '+str(curr_episode_total_reward)+' Total steps : '+str(steps_taken_per_episode))
        myBuffer.add(episodeBuffer.buffer)
        summary_val, = sess.run([curr_episode_reward_summary], feed_dict={curr_episode_total_reward_placeholder: curr_episode_total_reward})
        writer_op.add_summary(summary_val, episode_num + 1)
        reward_per_episode_list.append(curr_episode_total_reward)
        mean_reward_over_window = sum(reward_per_episode_list[-mean_reward_window_len:]) / min(len(reward_per_episode_list), mean_reward_window_len)
        summary_val, = sess.run([mean_reward_over_window_summary], feed_dict={mean_reward_over_window_placeholder: mean_reward_over_window})
        writer_op.add_summary(summary_val, episode_num + 1)
        mean_reward_per_episode_window_list.append(mean_reward_over_window)
        steps_taken_per_episode_list.append(steps_taken_per_episode)
        # Periodically save the model.
        if(episode_num % model_saving_freq == 0 and episode_num>0):
            saver.save(sess, path + '/model-' + str(episode_num) + '.ckpt')
            print("Saved Model after episode : "+str(episode_num))
        if len(reward_per_episode_list) % 10 == 0:
            print('Total steps taken till now, mean reward per episode, current epsilon :::::: ')
            print(str(total_steps)+', '+str(np.mean(reward_per_episode_list))+', '+str(e))
    saver.save(sess, path + '/model-' + str(episode_num) + '.ckpt')
    writer_op.close()
print("Percent of succesful episodes: " + str(sum(reward_per_episode_list) / num_episodes) + "%")

# ### Checking network learning
# Mean reward over time

reward_per_episode_list = np.array(reward_per_episode_list)
rMean = np.average(reward_per_episode_list)
print('Mean reward is '+str(rMean))

results_file_name = 'DQN_results.pickle'
fp = open(results_file_name, 'w')
results_dict = {'reward_per_episode_list':reward_per_episode_list, 'mean_reward_per_episode_window_list':mean_reward_per_episode_window_list, 'steps_taken_per_episode_list':steps_taken_per_episode_list}
pickle.dump(results_dict, fp)
fp.close()

plt.plot(reward_per_episode_list)

