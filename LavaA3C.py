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

# Tutorial sample #6: Discrete movement, rewards, and learning

# The "Cliff Walking" example using Q-learning.
# From pages 148-150 of:
# Richard S. Sutton and Andrews G. Barto
# Reinforcement Learning, An Introduction
# MIT Press, 1998

from future import standard_library
standard_library.install_aliases()
from builtins import range
from builtins import object
import MalmoPython
import json
import logging
import os
import random
import sys
import time
if sys.version_info[0] == 2:
    # Workaround for https://github.com/PythonCharmers/python-future/issues/262
    import Tkinter as tk
else:
    import tkinter as tk



import tensorflow as tf 
from tensorflow.contrib import slim 

import numpy as np 





learning_rate = 2e-3

video_width = 160
video_height = 100

batch_size = 100


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


class TabQAgent(object):
    """Tabular Q-learning agent for discrete state/action spaces."""

    def __init__(self):
        self.epsilon = 0.01 # chance of taking a random action instead of the best

        self.logger = logging.getLogger(__name__)
        if False: # True if you want to see more information
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

        self.actions = ["movenorth 1", "movesouth 1", "movewest 1", "moveeast 1"]
        self.q_table = {}
        self.canvas = None
        self.root = None

    def updateQTable( self, reward, current_state ):
        """Change q_table to reflect what we have learnt."""
        
        # retrieve the old action value from the Q-table (indexed by the previous state and the previous action)
        old_q = self.q_table[self.prev_s][self.prev_a]
        
        # TODO: what should the new action value be?
        new_q = old_q
        
        # assign the new action value to the Q-table
        self.q_table[self.prev_s][self.prev_a] = new_q
        
    def updateQTableFromTerminatingState( self, reward ):
        """Change q_table to reflect what we have learnt, after reaching a terminal state."""
        
        # retrieve the old action value from the Q-table (indexed by the previous state and the previous action)
        old_q = self.q_table[self.prev_s][self.prev_a]
        
        # TODO: what should the new action value be?
        new_q = old_q
        
        # assign the new action value to the Q-table
        self.q_table[self.prev_s][self.prev_a] = new_q
        
    def act(self, world_state, agent_host, current_r ):
        """take 1 action in response to the current world state"""
        
        obs_text = world_state.observations[-1].text
        obs = json.loads(obs_text) # most recent observation
        self.logger.debug(obs)
        if not u'XPos' in obs or not u'ZPos' in obs:
            self.logger.error("Incomplete observation received: %s" % obs_text)
            return 0
        current_s = "%d:%d" % (int(obs[u'XPos']), int(obs[u'ZPos']))
        self.logger.debug("State: %s (x = %.2f, z = %.2f)" % (current_s, float(obs[u'XPos']), float(obs[u'ZPos'])))
        if current_s not in self.q_table:
            self.q_table[current_s] = ([0] * len(self.actions))

        # update Q values
        if self.prev_s is not None and self.prev_a is not None:
            self.updateQTable( current_r, current_s )

        self.drawQ( curr_x = int(obs[u'XPos']), curr_y = int(obs[u'ZPos']) )

        # select the next action
        rnd = random.random()
        if rnd < self.epsilon:
            a = random.randint(0, len(self.actions) - 1)
            self.logger.info("Random action: %s" % self.actions[a])
        else:
            m = max(self.q_table[current_s])
            self.logger.debug("Current values: %s" % ",".join(str(x) for x in self.q_table[current_s]))
            l = list()
            for x in range(0, len(self.actions)):
                if self.q_table[current_s][x] == m:
                    l.append(x)
            y = random.randint(0, len(l)-1)
            a = l[y]
            self.logger.info("Taking q action: %s" % self.actions[a])

        # try to send the selected action, only update prev_s if this succeeds
        try:
            agent_host.sendCommand(self.actions[a])
            self.prev_s = current_s
            self.prev_a = a

        except RuntimeError as e:
            self.logger.error("Failed to send command: %s" % e)

        return current_r

    def run(self, agent_host):
        """run the agent on the world"""

        total_reward = 0
        
        self.prev_s = None
        self.prev_a = None
        
        is_first_action = True
        
        # main loop:
        world_state = agent_host.getWorldState()
        while world_state.is_mission_running:

            current_r = 0
            
            if is_first_action:
                # wait until have received a valid observation
                while True:
                    time.sleep(0.1)
                    world_state = agent_host.getWorldState()
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        current_r += reward.getValue()
                    if world_state.is_mission_running and len(world_state.observations)>0 and not world_state.observations[-1].text=="{}":
                        total_reward += self.act(world_state, agent_host, current_r)
                        break
                    if not world_state.is_mission_running:
                        break
                is_first_action = False
            else:
                # wait for non-zero reward
                while world_state.is_mission_running and current_r == 0:
                    time.sleep(0.1)
                    world_state = agent_host.getWorldState()
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        current_r += reward.getValue()
                # allow time to stabilise after action
                while True:
                    time.sleep(0.1)
                    world_state = agent_host.getWorldState()
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        current_r += reward.getValue()
                    if world_state.is_mission_running and len(world_state.observations)>0 and not world_state.observations[-1].text=="{}":
                        total_reward += self.act(world_state, agent_host, current_r)
                        break
                    if not world_state.is_mission_running:
                        break

        # process final reward
        self.logger.debug("Final reward: %d" % current_r)
        total_reward += current_r

        # update Q values
        if self.prev_s is not None and self.prev_a is not None:
            self.updateQTableFromTerminatingState( current_r )
            
        self.drawQ()
    
        return total_reward
        
    def drawQ( self, curr_x=None, curr_y=None ):
        scale = 40
        world_x = 6
        world_y = 14
        if self.canvas is None or self.root is None:
            self.root = tk.Tk()
            self.root.wm_title("Q-table")
            self.canvas = tk.Canvas(self.root, width=world_x*scale, height=world_y*scale, borderwidth=0, highlightthickness=0, bg="black")
            self.canvas.grid()
            self.root.update()
        self.canvas.delete("all")
        action_inset = 0.1
        action_radius = 0.1
        curr_radius = 0.2
        action_positions = [ ( 0.5, action_inset ), ( 0.5, 1-action_inset ), ( action_inset, 0.5 ), ( 1-action_inset, 0.5 ) ]
        # (NSWE to match action order)
        min_value = -20
        max_value = 20
        for x in range(world_x):
            for y in range(world_y):
                s = "%d:%d" % (x,y)
                self.canvas.create_rectangle( x*scale, y*scale, (x+1)*scale, (y+1)*scale, outline="#fff", fill="#000")
                for action in range(4):
                    if not s in self.q_table:
                        continue
                    value = self.q_table[s][action]
                    color = 255 * ( value - min_value ) / ( max_value - min_value ) # map value to 0-255
                    color = max( min( color, 255 ), 0 ) # ensure within [0,255]
                    color_string = '#%02x%02x%02x' % (255-color, color, 0)
                    self.canvas.create_oval( (x + action_positions[action][0] - action_radius ) *scale,
                                             (y + action_positions[action][1] - action_radius ) *scale,
                                             (x + action_positions[action][0] + action_radius ) *scale,
                                             (y + action_positions[action][1] + action_radius ) *scale, 
                                             outline=color_string, fill=color_string )
        if curr_x is not None and curr_y is not None:
            self.canvas.create_oval( (curr_x + 0.5 - curr_radius ) * scale, 
                                     (curr_y + 0.5 - curr_radius ) * scale, 
                                     (curr_x + 0.5 + curr_radius ) * scale, 
                                     (curr_y + 0.5 + curr_radius ) * scale, 
                                     outline="#fff", fill="#fff" )
        self.root.update()





class A3C:
    def __init__(self,gamma=0.96):
        
        self.action_space = ['move 1', 'move -1', 'turn 1', 'turn -1']

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

    def run(self, agent_host,epsilon=0.3):

        episode_buffer = []
        episode_values = []
        episode_frames = []
        episode_reward = 0

        rnn_state = self.state_init
            
        world_state = agent_host.getWorldState()

        while (len(world_state.video_frames) == 0):
            time.sleep(0.1)
            world_state = agent_host.getWorldState()

        s = process_pixels(world_state.video_frames[0].pixels)

        while(world_state.is_mission_running):

            a_dist,v,rnn_state = self.sess.run([self.policy,self.value,self.state_out], 
                        feed_dict={self.inputs:[s],
                        self.state_in[0]:rnn_state[0],
                        self.state_in[1]:rnn_state[1]})
            a_greedy = np.argmax(a_dist)
            if (random.random > epsilon):
                a = a_greedy
            else :
                a = random.randint(0,len(self.action_space))
        
            agent_host.sendCommand(self.action_space[a])
            time.sleep(0.05)

            world_state = agent_host.getWorldState()
        
            #r = world_state.rewards[0].getValue()
            r = get_rewards(world_state.rewards)
            
            d = world_state.is_mission_running

            if (d == False):
                try:
                    world_state = agent_host.getWorldState()
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
        #self.episode_reward.append(episode_reward)
        v_l,p_l,e_l = self.train(episode_buffer,self.sess,self.gamma,0.0)

        
        self.episode_count += 1

        if (self.episode_count % 20 == 0):
            self.saver.save(self.sess,'maze_models/maze_ep_{0}.cpkt'.format(self.episode_count))


        print('episode {0} buffer = {4}, value loss = {1} | policy = {2} | entropy = {3}'.format(self.episode_count,v_l,p_l,e_l,len(episode_buffer)))
        episode_buffer = []
        
        return episode_reward 









if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)

agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse( sys.argv )
except RuntimeError as e:
    print('ERROR:',e)
    print(agent_host.getUsage())
    exit(1)
if agent_host.receivedArgument("help"):
    print(agent_host.getUsage())
    exit(0)

# -- set up the mission -- #
mission_file = './LavaA3C.xml'
with open(mission_file, 'r') as f:
    print("Loading mission from %s" % mission_file)
    mission_xml = f.read()
    my_mission = MalmoPython.MissionSpec(mission_xml, True)
# add 20% holes for interest
for x in range(1,4):
    for z in range(1,13):
        if random.random()<0.1:
            my_mission.drawBlock( x,45,z,"lava")

max_retries = 3

if agent_host.receivedArgument("test"):
    num_repeats = 1
else:
    num_repeats = 150



agent = A3C()

cumulative_rewards = []
for i in range(num_repeats):

    print()
    print('Repeat %d of %d' % ( i+1, num_repeats ))
    
    my_mission_record = MalmoPython.MissionRecordSpec()

    for retry in range(max_retries):
        try:
            agent_host.startMission( my_mission, my_mission_record )
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:",e)
                exit(1)
            else:
                time.sleep(2.5)

    print("Waiting for the mission to start", end=' ')
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        print(".", end="")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)
    print()

    # -- run the agent in the world -- #
    cumulative_reward = agent.run(agent_host)
    print('Cumulative reward: %d' % cumulative_reward)
    cumulative_rewards += [ cumulative_reward ]

    # -- clean up -- #
    time.sleep(1) # (let the Mod reset)

print("Done.")

print()
print("Cumulative rewards for all %d runs:" % num_repeats)
print(cumulative_rewards)
