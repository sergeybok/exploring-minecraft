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

# Tutorial sample #2: Run simple mission using raw XML

#from builtins import range
import MalmoPython
import os
import sys
import time
import random
import json
import numpy as np

import cv2

if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)

# More interesting generator string: "3;7,44*49,73,35:1,159:4,95:13,35:13,159:11,95:10,159:14,159:6,35:6,95:6;12;"

video_width = 432
video_height = 240

missionXML='''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
            
              <About>
                <Summary>Hello world!</Summary>
              </About>

              
              <ServerSection>

                <ServerInitialConditions>
                  <Weather>clear</Weather>
                </ServerInitialConditions>

                <ServerHandlers>
                  <DefaultWorldGenerator />
                  <ServerQuitFromTimeUp timeLimitMs="5000"/>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>
              
              <AgentSection mode="Survival">
                <Name>MalmoTutorialBot</Name>
                <AgentStart/>
                <AgentHandlers>
                    <VideoProducer want_depth="true">
                        <Width>''' + str(video_width) + '''</Width>
                        <Height>''' + str(video_height) + '''</Height>
                    </VideoProducer>
                  <ObservationFromFullStats/>
                  <ContinuousMovementCommands turnSpeedDegs="180"/>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''

# Create default Malmo objects:

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


agent_host.setVideoPolicy(MalmoPython.VideoPolicy.KEEP_ALL_FRAMES)

my_mission = MalmoPython.MissionSpec(missionXML, True)
my_mission_record = MalmoPython.MissionRecordSpec()

# Attempt to start a mission:
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

# Loop until mission starts:
print("Waiting for the mission to start ", end=' ')
world_state = agent_host.getWorldState()
while not world_state.has_mission_begun:
    print(".", end="")
    time.sleep(0.1)
    world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print("Error:",error.text)

print()
print("Mission running ", end=' ')

action_space = ['move 1',
                'move -1',
                'strafe -1',
                'strafe 1',
                'pitch 0.1'
                'pitch -0.1'
                'pitch 0'
                'turn 1',
                'turn -1',
                'turn 0',
                'jump 1',
                'jump 0',
                'crouch 1',
                'crouch 0',
                'attack 1',
                'attack 0',
                'use 1',
                'use 0']


world_state = agent_host.getWorldState()
while len(world_state.observations) == 0:
  time.sleep(0.1)
  world_state = agent_host.getWorldState()


frame_dir = 'frames/'
if not os.path.exists(frame_dir):
  os.makedirs(frame_dir)

# Loop until mission ends:
i = 0
while world_state.is_mission_running:
    if len(world_state.video_frames) == 0:
      time.sleep(0.05)
      world_state = agent_host.getWorldState()
      agent_host.sendCommand(random.choice(action_space))
      print('The length of video frames is' + str(len(world_state.video_frames)))
      continue

    print('After the continue statement the length of video frames is'+str(len(world_state.video_frames)))

    i+=1
    print(".", end="")
    time.sleep(0.1)
    world_state = agent_host.getWorldState()
    #print(world_state.__dict__)
    msg = world_state.observations[-1].text
    curr_observation = json.loads(msg)
    if i % 20 == 0:
      frame = world_state.video_frames[0]
      img = np.array(cv2.imdecode((frame.pixels), cv2.CV_LOAD_IMAGE_COLOR))
      cv2.imwrite('{0}frame_{1}.jpg'.format(frame_dir,i),im)

    print(curr_observation)
    print('')
    for error in world_state.errors:
        print("Error:",error.text)
    agent_host.sendCommand(random.choice(action_space))


print()
print("Mission ended")
# Mission has ended.
