import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import os

from environments import maze_environment

class Qnetwork():
    def __init__(self, frame_height=None, frame_width=None, frame_channels=None, network_name=None):
        # The network recieves a frame from the game, flattened into an array.
        # It then resizes it and processes it through four convolutional layers.
        # TODO :: DONE define the frame size here
        flattened_frame_size = frame_height*frame_width*frame_channels
        num_final_layer_output_channel = 512
        total_num_actions = None
        self.flattened_image = tf.placeholder(shape=[None, flattened_frame_size], dtype=tf.float32)
        #[batch, in_height, in_width, in_channels]
        #[filter_height, filter_width, in_channels, out_channels]
        # Must have strides[0] = strides[3] = 1. For the most common case of the same horizontal and vertices strides, strides = [1, stride, stride, 1].
        self.reshaped_image = tf.reshape(self.flattened_image, shape=[-1, frame_height, frame_width, frame_channels])
        self.conv1 = self.conv_layer(input_volume=self.reshaped_image, num_output_channel=32, filter_shape=[8, 8], strides_shape=[1, 4, 4, 1], padding_type='VALID', network_name=network_name, layer_name='1')
        self.conv2 = self.conv_layer(input_volume=self.conv1, num_output_channel=64, filter_shape=[4, 4], strides_shape=[1, 2, 2, 1], padding_type='VALID', network_name=network_name, layer_name='2')
        self.conv3 = self.conv_layer(input_volume=self.conv2, num_output_channel=128, filter_shape=[3, 3], strides_shape=[1, 1, 1, 1], padding_type='VALID', network_name=network_name, layer_name='3')
        self.conv4 = self.conv_layer(input_volume=self.conv3, num_output_channel=512, filter_shape=[7, 7], strides_shape=[1, 1, 1, 1], padding_type='VALID', network_name=network_name, layer_name='4')

        #TODO Figure out why is the split being done this way, why is split required at all???
        # We take the output from the final convolutional layer and split it into separate advantage and value streams.
        self.streamAC, self.streamVC = tf.split(self.conv4, 2, 3)
        self.streamA = tf.contrib.layers.flatten(self.streamAC)
        self.streamV = tf.contrib.layers.flatten(self.streamVC)
        xavier_init = tf.contrib.layers.xavier_initializer()
        #TODO add bias to the advantage and value functions
        self.AW = tf.Variable(xavier_init([num_final_layer_output_channel // 2, total_num_actions]))
        self.VW = tf.Variable(xavier_init([num_final_layer_output_channel // 2, 1]))
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)

        # Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
        self.predict = tf.argmax(self.Qout, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, total_num_actions, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)


    def conv_layer(self, input_volume = None, num_output_channel = None, filter_shape = None, strides_shape = None, padding_type = None, network_name=None, layer_name = None,):
        std_dev = 0.1
        W_shape = [filter_shape[0], filter_shape[1], input_volume.shape[3].value, num_output_channel]
        W = tf.Variable(tf.truncated_normal(W_shape, stddev=std_dev), name=network_name+'_Filter_'+layer_name)
        b_shape = [num_output_channel]
        b = tf.Variable(tf.constant(0.1, shape=b_shape), name=network_name+'_bias_'+layer_name)
        conv = tf.nn.conv2d(input_volume, W, strides_shape, padding=padding_type, name=network_name+'_Conv_'+layer_name)
        conv = conv + b
        relu = tf.nn.relu(conv, name=network_name+'_ReLU_'+layer_name)
        return relu
        # pool1_ksize = [1, 2, 2, 1]
        # strides_pool1 = [1, 2, 2, 1]
        # max_pool1 = tf.nn.max_pool(relu1, pool1_ksize, strides_pool1, padding='SAME', name='Pool1')

# ### Experience Replay
# This class allows us to store experiences and sample then randomly to train the network.
class experience_buffer():
    def __init__(self, buffer_size=50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        #Since we have 5 elements in the experience : s, a, r, s1, is_terminal_flag, therefore we reshape it as [size, 5]
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])


# # This is a simple function to resize our game frames.
# def processState(states):
#     return np.reshape(states, [21168])


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
batch_size = 32  # How many experiences to use for each training step.
update_freq = 10  # How often to perform a training step.
gamma_discount_factor = .99  # Discount factor on the target Q-values
startE = 1  # Starting chance of random action
endE = 0.1  # Final chance of random action
annealing_steps = 5000.  # How many steps of training to reduce startE to endE.
num_episodes = 5000  # How many episodes of game environment to train network with.
pre_train_steps = 5000  # How many steps of random actions before training begins.
max_epLength = 50  # The max allowed length of our episode.
load_model = False  # Whether to load a saved model.
path = "./dqn_model"  # The path to save our model to.
# h_size = 512  # The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001  # Rate to update target network toward primary network

#TODO why is reset default graph required??
tf.reset_default_graph()
maze_env = maze_environment.environment()
frame_height = maze_env.video_height
frame_width = maze_env.video_width
frame_channels = maze_env.video_channels
mainQN = Qnetwork(frame_height=frame_height, frame_width=frame_width, frame_channels=frame_channels, network_name='main')
targetQN = Qnetwork(frame_height=frame_height, frame_width=frame_width, frame_channels=frame_channels, network_name='target')

init = tf.global_variables_initializer()

saver = tf.train.Saver()

trainables = tf.trainable_variables()

targetOps = updateTargetGraph(trainables, tau)

myBuffer = experience_buffer()

# Set the rate of random action decrease.
e = startE
stepDrop = (startE - endE) / annealing_steps

# create lists to contain total rewards and steps per episode
jList = []
rList = []
total_steps = 0

# Make a path for our model to be saved in.
if not os.path.exists(path):
    os.makedirs(path)

with tf.Session() as sess:
    sess.run(init)
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    for i in range(num_episodes):
        maze_env.get_maze()
        episodeBuffer = experience_buffer()
        # Reset environment and get first new observation
        s = maze_env.get_current_state()
        #TODO define total_num_actions, also define whether continous or discrete actions
        total_num_actions = maze_env.total_num_actions
        # s = processState(s)
        is_terminal_flag = False
        rAll = 0
        j = 0
        # The Q-Network
        while j < max_epLength:  # If the agent takes longer than 50 moves to reach the end of the maze, end the trial.
            j += 1
            # Choose an action by greedily (with e chance of random action) from the Q-network
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                a = np.random.randint(0, total_num_actions)
            else:
                a = sess.run(mainQN.predict, feed_dict={mainQN.flattened_image: [s]})[0]
            s1, r, is_terminal_flag = maze_env.take_action(a)
            # s1 = processState(s1)
            total_steps += 1
            episodeBuffer.add(
                np.reshape(np.array([s, a, r, s1, is_terminal_flag]), [1, 5]))  # Save the experience to our episode buffer.
            # Since we have 5 elements in the experience : s, a, r, s1, is_terminal_flag, therefore we reshape it as [size, 5]

            if total_steps > pre_train_steps:
                #NOTE::: We are reducing the epsilon of exploration after every action we take, not after every episode, so the epsilon decreases within 1 episode
                if e > endE:
                    e -= stepDrop

                if total_steps % (update_freq) == 0:
                    trainBatch = myBuffer.sample(batch_size)  # Get a random batch of experiences.
                    # Below we perform the Double-DQN update to the target Q-values
                    Q1 = sess.run(mainQN.predict, feed_dict={mainQN.flattened_image: np.vstack(trainBatch[:, 3])})
                    Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.flattened_image: np.vstack(trainBatch[:, 3])})
                    #TODO :: Done figure out the use of end_multiplier???, the is_terminal_flag gets stored as 1 or 0(True or False),
                    #todo :: Done if the is_terminal_flag is true i.e 1, we define the end_multiplier as 0, if the is_terminal_flag is false i.e 0, we define the end_multiplier as 1
                    end_multiplier = -(trainBatch[:, 4] - 1)
                    doubleQ = Q2[range(batch_size), Q1]
                    targetQ = trainBatch[:, 2] + (gamma_discount_factor * doubleQ * end_multiplier)
                    # Update the network with our target values.
                    #TODO it is important to recalculate the Q values of the states in the experience replay and then get the gradient w.r.t difference b/w recalculated values and targets
                    #TODO otherwise it defeats the purpose of experience replay, also we are not storing the Q values for this reason
                    _ = sess.run(mainQN.updateModel,
                                 feed_dict={mainQN.flattened_image: np.vstack(trainBatch[:, 0]), mainQN.targetQ: targetQ,
                                            mainQN.actions: trainBatch[:, 1]})

                    updateTarget(targetOps, sess)  # Update the target network toward the primary network.
            rAll += r
            s = s1

            if is_terminal_flag == True:
                break

        myBuffer.add(episodeBuffer.buffer)
        jList.append(j)
        rList.append(rAll)
        # Periodically save the model.
        if i % 1000 == 0:
            saver.save(sess, path + '/model-' + str(i) + '.ckpt')
            print("Saved Model")
        if len(rList) % 10 == 0:
            print(total_steps, np.mean(rList[-10:]), e)
    saver.save(sess, path + '/model-' + str(i) + '.ckpt')
print("Percent of succesful episodes: " + str(sum(rList) / num_episodes) + "%")

# ### Checking network learning
# Mean reward over time

rMat = np.resize(np.array(rList), [len(rList) // 100, 100])
rMean = np.average(rMat, 1)
plt.plot(rMean)

