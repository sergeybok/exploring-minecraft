from __future__ import print_function
# ------------------------------------------------------------------------------------------------
# Copyright (c) 2016 Microsoft Corporation
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------

from builtins import range
import MalmoPython
import os
import random
import sys
import time
import json
import errno

import tensorflow as tf 
from tensorflow.contrib import slim 
import scipy
import numpy as np 

from threading import Thread, Lock


learning_rate = 1e-3

video_width = 160
video_height = 100

batch_size = 100




maze4 = '''
    <MazeDecorator>
        <SizeAndPosition length="15" width="15" yOrigin="225" zOrigin="0" height="180"/>
        <GapProbability variance="0.4">0.4</GapProbability>
        <Seed>123</Seed>
        <MaterialSeed>124</MaterialSeed>
        <AllowDiagonalMovement>false</AllowDiagonalMovement>
        <StartBlock fixedToEdge="true" type="emerald_block" height="1"/>
        <EndBlock fixedToEdge="true" type="redstone_block" height="12"/>
        <PathBlock type="dirt" height="1"/>
        <FloorBlock type="stone" variant="smooth_granite"/>
        <SubgoalBlock type="glowstone"/>
        <OptimalPathBlock type="stone" variant="smooth_diorite"/>
        <GapBlock type="air" height="1"/>
        <AddQuitProducer description="finished maze"/>
        <AddNavigationObservations/>
    </MazeDecorator>
'''

def GetMissionXML( mazeblock ):
    return '''<?xml version="1.0" encoding="UTF-8" ?>
    <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
        <About>
            <Summary>Run the maze!</Summary>
        </About>
        
        <ModSettings>
            <MsPerTick>''' + str(TICK_LENGTH) + '''</MsPerTick>
        </ModSettings>

        <ServerSection>
            <ServerInitialConditions>
                <AllowSpawning>false</AllowSpawning>
            </ServerInitialConditions>
            <ServerHandlers>
                <FlatWorldGenerator generatorString="3;7,220*1,5*3,2;3;,biome_1" />
                ''' + mazeblock + '''
                <ServerQuitFromTimeUp timeLimitMs="45000"/>
                <ServerQuitWhenAnyAgentFinishes />
            </ServerHandlers>
        </ServerSection>

        <AgentSection mode="Survival">
            <Name>James Bond</Name>
            <AgentStart>
                <Placement x="-204" y="81" z="217"/>
            </AgentStart>
            <AgentHandlers>
                <RewardForTouchingBlockType>
                    <Block reward="100.0" type="redstone_block" behaviour="onceOnly"/>
                    <Block reward="20.0" type="glowstone"/>
                    <Block reward="10.0" type="stone" variant="smooth_diorite"/>
                </RewardForTouchingBlockType>
                <VideoProducer want_depth="false">
                    <Width>''' + str(video_width) + '''</Width>
                    <Height>''' + str(video_height) + '''</Height>
                </VideoProducer>
            </AgentHandlers>
        </AgentSection>

    </Mission>'''


def discount(x, gamma):
    g = 1
    for i in range(x.shape[0]):
        x[i] *=  g
        g *= g 
    return x


def process_pixels(pixels):
    img = np.frombuffer(pixels, dtype=np.uint8)
    img = img.reshape((video_height,video_width,3)).astype(np.float32) / 255.0
    return img

def get_rewards(rewards):
    r = 0
    for reward in rewards:
        r += reward.getValue()

    return r


class A3C:
    def __init__(self, agent_host, action_space,gamma=0.96):
        self.agent_host = agent_host
        if action_space == 'discrete relative' or True:
            self.action_space = ['move 1', 'move -1', 'turn 1', 'turn -1']
        else:
            self.action_space = ['movenorth 1', 'movesouth 1', 'movewest 1', 'moveeast 1'] 


        self.gamma = gamma

        self.episode_count = 0

        self.sess = tf.InteractiveSession()

        self.init_network()

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        self.obs = []
        self.episode_rewards = []


    def init_network(self):

        self.inputs = tf.placeholder(tf.float32,shape=[None,video_height,video_width,3],name='inputs')



        self.conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                            inputs=self.inputs,num_outputs=12,
                            kernel_size=[8,8],stride=[4,4],
                            padding='VALID',scope='conv1')
        self.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
                            inputs=self.conv1,num_outputs=20,
                            kernel_size=[4,4],stride=[2,2],
                            padding='VALID',scope='conv2')
        self.pool = slim.max_pool2d(self.conv2,[4,4])
        hidden = slim.fully_connected(slim.flatten(self.pool),256,activation_fn=tf.nn.elu,scope='fc_out')

        self.lstm_cell = lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(256,state_is_tuple=True)
        c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
        h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c],name='cin')
        h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h],name='hin')
        self.state_in = (c_in, h_in)
        rnn_in = tf.expand_dims(hidden, [0])
        step_size = tf.shape(self.inputs)[:1]
        state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                time_major=False)

        lstm_c, lstm_h = lstm_state
        self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
        rnn_out = tf.reshape(lstm_outputs, [-1, 256])

        self.policy = slim.fully_connected(rnn_out,len(self.action_space),
                activation_fn=tf.nn.softmax,scope='policy_fc')        
        self.value = slim.fully_connected(rnn_out,1,
                activation_fn=None,scope='value_fc')

        self.actions_taken = tf.placeholder(shape=[None],dtype=tf.int32,name='actions_taken')
        self.actions_onehot = tf.one_hot(self.actions_taken,len(self.action_space),dtype=tf.float32,name='actions_hot')
        self.target_v = tf.placeholder(shape=[None],dtype=tf.float32,name='target_v')
        self.advantages = tf.placeholder(shape=[None],dtype=tf.float32,name='advantages')

        self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

        self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])),name='value_loss')
        self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy),name='entropy_loss')
        self.policy_loss = tf.abs(tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages,name='policy_loss'))
        self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

        optimizer = tf.train.AdamOptimizer(learning_rate)

        self.gradients = optimizer.compute_gradients(self.loss)

        for g in self.gradients:
            tf.clip_by_value(g,-20,20)

        self.train_step = optimizer.apply_gradients(self.gradients)

    def train(self, rollout, sess, gamma, bootstrap_value):
        #rollout = np.array(rollout)
        observations = []
        actions = []
        rewards = []
        next_observations = []
        values = []
        for i in range(len(rollout)):
            observations.append(rollout[i][0])
            actions.append(rollout[i][1])
            rewards.append(rollout[i][2])
            next_observations.append(rollout[i][3])
            values.append(rollout[i][-1])

        observations = np.array(observations)
        print('obs shape {0}'.format(observations.shape))
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_observations = np.array(next_observations)
        values = np.array(values)

        #observations = observations.reshape((observations.shape[0]//(video_height*video_width),video_height,video_width))
        #actions = rollout[:,1]
        #rewards = rollout[:,2]
        #next_observations = rollout[:,3]
        #values = rollout[:,5]

        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages,gamma)

        #print('obs shape {0}'.format(observations.shape))
        n_batches = len(rollout)//batch_size
        rem = len(rollout) % batch_size

        avg_vl = 0 
        avg_pl = 0
        avg_el = 0

        for b in range(n_batches):
            obs_in = observations[b*batch_size:(b+1)*batch_size]
            #print('obs in type {0}'.format(obs_in.dtype))
            #obs_in = np.zeros((100,video_height,video_width,3))
            feed_dict = {self.target_v:discounted_rewards[b*batch_size:(b+1)*batch_size],
                    self.inputs:obs_in,
                    self.actions_taken:actions[b*batch_size:(b+1)*batch_size],
                    self.advantages:advantages[b*batch_size:(b+1)*batch_size],
                    self.state_in: (np.zeros((1,256,)),np.zeros((1,256)))
                    }
            _,v_l,p_l,e_l = self.sess.run([self.train_step,
                    self.value_loss,self.policy_loss, self.entropy],
                    feed_dict=feed_dict)
            avg_vl += v_l.mean()/batch_size
            avg_pl += p_l.mean()/batch_size
            avg_el += e_l.mean()/batch_size

        feed_dict = {self.target_v:discounted_rewards[-rem:],
                    self.inputs:observations[-rem:],
                    self.actions_taken:actions[-rem:],
                    self.advantages:advantages[-rem:],
                    self.state_in:(np.zeros((1,256,)),np.zeros((1,256)))
                    }
        _,v_l,p_l,e_l = self.sess.run([self.train_step,
                    self.value_loss,self.policy_loss, self.entropy],feed_dict=feed_dict)
        avg_vl += v_l.mean()/batch_size
        avg_pl += p_l.mean()/batch_size
        avg_el += e_l.mean()/batch_size

        return avg_vl, avg_pl, avg_el



    def add_obs(self,cur_frame):
        self.obs.append(cur_frame)

    def add_reward(self, cur_reward):
        self.rewards.append(cur_reward)


    def run(self):

        episode_buffer = []
        episode_values = []
        episode_frames = []
        episode_reward = 0

        rnn_state = self.state_init
            
        world_state = self.agent_host.getWorldState()

        while (len(world_state.video_frames) == 0):
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()

        s = process_pixels(world_state.video_frames[0].pixels)

        while(world_state.is_mission_running):

            a_dist,v,rnn_state = self.sess.run([self.policy,self.value,self.state_out], 
                        feed_dict={self.inputs:[s],
                        self.state_in[0]:rnn_state[0],
                        self.state_in[1]:rnn_state[1]})
            a = np.random.choice(a_dist[0],p=a_dist[0])
            a = np.argmax(a_dist == a)
        
            self.agent_host.sendCommand(self.action_space[a])

            time.sleep(0.05)
        
            #r = world_state.rewards[0].getValue()
            r = get_rewards(world_state.rewards)
            
            d = world_state.is_mission_running

            if (d == False):
                try:
                    world_state = self.agent_host.getWorldState()
                    s1 = process_pixels(world_state.video_frames[0].pixels)
                    episode_frames.append(s1)
                except:
                    s1 = s
            else :
                s1 = s


            episode_buffer.append([s,a,r,s1,d,v[0,0]])
            episode_values.append(v[0,0])

            episode_reward += r
            s = s1 

            world_state = self.agent_host.getWorldState()

        #self.episode_reward.append(episode_reward)
        v_l,p_l,e_l = self.train(episode_buffer,self.sess,self.gamma,0.0)

        episode_buffer = []
        self.episode_count += 1

        if (self.episode_count % 20 == 0):
            self.saver.save(self.sess,'maze_models/maze_ep_{0}.cpkt'.format(self.episode_count))


        print('episode reward = {0}'.format(episode_reward))
        print('episode {0}, value loss = {1} | policy = {2} | entropy = {3}'.format(self.episode_count,v_l,p_l,e_l))







  
if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)

validate = True
#mazeblocks = [maze1, maze2, maze3, maze4]

mazeblocks = [maze4]


agent_host = MalmoPython.AgentHost()
agent_host.addOptionalIntArgument( "speed,s", "Length of tick, in ms.", 10)
try:
    agent_host.parse( sys.argv )
except RuntimeError as e:
    print('ERROR:',e)
    print(agent_host.getUsage())
    exit(1)
if agent_host.receivedArgument("help"):
    print(agent_host.getUsage())
    exit(0)

#agent_host.setObservationsPolicy(MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY)
agent_host.setVideoPolicy(MalmoPython.VideoPolicy.LATEST_FRAME_ONLY)


if agent_host.receivedArgument("test"):
    num_reps = 1
else:
    num_reps = 30000

recordingsDirectory="MazeRecordings"
TICK_LENGTH = agent_host.getIntArgument("speed")

try:
    os.makedirs(recordingsDirectory)
except OSError as exception:
    if exception.errno != errno.EEXIST: # ignore error if already existed
        raise

# Set up a recording
my_mission_record = MalmoPython.MissionRecordSpec()
my_mission_record.recordRewards()
my_mission_record.recordObservations()




# init agent 



agent = A3C(agent_host,action_space='discrete relative')




for iRepeat in range(num_reps):
    my_mission_record.setDestination(recordingsDirectory + "//" + "Mission_" + str(iRepeat) + ".tgz")
    mazeblock = random.choice(mazeblocks)
    my_mission = MalmoPython.MissionSpec(GetMissionXML(mazeblock),validate)
    
    my_mission.allowAllDiscreteMovementCommands()

    max_retries = 3
    for retry in range(max_retries):
        try:
            agent_host.startMission( my_mission, my_mission_record )
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:",e)
                exit(1)
            else:
                time.sleep(2)

    print("Waiting for the mission to start", end=' ')
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        print(".", end="")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        if len(world_state.errors):
            print()
            for error in world_state.errors:
                print("Error:",error.text)
                exit()
    print()

    # initialize a3c agent

    agent.run()
                
    print("Mission has stopped.")
    time.sleep(0.5) # Give mod a little time to get back to dormant state.
