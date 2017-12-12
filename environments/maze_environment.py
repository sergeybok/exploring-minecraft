import MalmoPython
import numpy as np
import os

import random
import sys
import time
import json
import errno

class environment():
    def __init__(self):
        self.maze4 = '''
            <MazeDecorator>
                <SizeAndPosition length="10" width="10" yOrigin="5" zOrigin="5" height="15"/>
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
                <GapBlock type="dirt" height="1"/>
                <AddQuitProducer description="finished maze"/>
                <AddNavigationObservations/>
            </MazeDecorator>
            <DrawingDecorator>
              <DrawBlock type="lapis_block" y="5" z="5" x="15"/>
            </DrawingDecorator>
        '''

        # <SizeAndPosition length = "10" width = "10" yOrigin = "225" zOrigin = "0" height = "180"/>

        # TODO: to make it easier for the agent i have now defined the gap block as dirt, therefore basically there are no gaps
        # TODO: in a more advanced version we will use air as the gap block in this version the agent will have to take the 'jump' action to get out of the vacant hole (i.e air gap block)

        # NOTE:::: OptimalPathBlock is the optimal path hints to the final goal, these are the stones which connect the starting position to the final goal positions via the subgoals
        # SubgoalBlock are the stones which define the sub goals along the way to the final goal
        # GapProbability takes value between 0 and 2, not from 0 to 1. It is the probability of having a 'gap'/hole in the elevated waliking area, it not only changes the total number of availabale blocks to walk on,
        # GapProbability also changes the optimal path to the final goal. For value 0.0 there are no gaps in the walking area, but the agent does not follow the given optimal path which is two perpendicular paths
        # but instead walks in a diagonal shortest path to the goal. For value 2.0 there are no walking blocks except for the optimal path, i.e the floor is all empty but for the optimal path and the optimal path
        # is a straight line to the final goal and the agent follows this optimal path. The gap block doesn't necessarily have to be air, it can be defined as lapis_ore or anything else as well.

        # self.video_width = 256
        self.video_width = 84
        # self.video_height = 256
        self.video_height = 84
        self.want_depth_channel = 'false'
        if(self.want_depth_channel=='false'):
            self.video_channels = 3
        elif(self.want_depth_channel=='true'):
            self.video_channels = 4
        self.terminal_state_visit_count = 0

        sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately

        self.validate = True
        self.mazeblocks = [self.maze4]

        self.agent_host = MalmoPython.AgentHost()
        self.agent_host.addOptionalIntArgument("speed,s", "Length of tick, in ms.", 50)
        try:
            self.agent_host.parse(sys.argv)
        except RuntimeError as e:
            print('ERROR:')
            print(e)
            print('\n')
            self.agent_host.getUsage()
            exit(1)
        if self.agent_host.receivedArgument("help"):
            print(self.agent_host.getUsage())
            exit(0)

        self.agent_host.setObservationsPolicy(MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY)
        self.agent_host.setRewardsPolicy(MalmoPython.RewardsPolicy.KEEP_ALL_REWARDS)
        self.agent_host.setVideoPolicy(MalmoPython.VideoPolicy.LATEST_FRAME_ONLY)

        self.recordingsDirectory = "MazeRecordings"
        self.TICK_LENGTH = self.agent_host.getIntArgument("speed")

        try:
            os.makedirs(self.recordingsDirectory)
        except OSError as exception:
            if exception.errno != errno.EEXIST:  # ignore error if already existed
                raise

        self.curr_episode_num = 0
        # self.action_dict = {0: 'move 0.3', 1: 'move 0', 2: 'move -0.3', 3: 'turn 0.1', 4: 'turn 0', 5: 'turn -0.1'}
        self.action_dict = {0:"movenorth 1", 1:"movesouth 1", 2:"movewest 1", 3:"moveeast 1"}
        # self.action_dict = {0:['move 0.3', 'move 0'], 1:['move -0.3', 'move 0'], 2:['turn 0.1', 'turn 0'], 3:['turn -0.1', 'turn 0']}
        self.total_num_actions = len(self.action_dict)

    def get_maze(self):
        # Set up a recording
        my_mission_record = MalmoPython.MissionRecordSpec()
        my_mission_record.recordRewards()
        my_mission_record.recordObservations()

        my_mission_record.setDestination(self.recordingsDirectory + "//" + "Mission_" + str(self.curr_episode_num) + ".tgz")
        mazeblock = random.choice(self.mazeblocks)
        my_mission = MalmoPython.MissionSpec(self.GetMissionXML(mazeblock), self.validate)

        max_retries = 3
        for retry in range(max_retries):
            try:
                self.agent_host.startMission(my_mission, my_mission_record)
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:")
                    print(e)
                    exit(1)
                else:
                    time.sleep(2)

        print("Waiting for the mission to start")
        self.world_state = self.agent_host.getWorldState()
        while not self.world_state.has_mission_begun:
            sys.stdout.write(".")
            time.sleep(0.1)
            self.world_state = self.agent_host.getWorldState()
            if len(self.world_state.errors):
                print('\n')
                for error in self.world_state.errors:
                    print("Error:"+error.text)
                    exit()
        print('\nMission has started')
        self.curr_episode_num += 1

    # TODO :: fix the reward definition at the end of maze i.e touching the redstone properly
    # TODO :: cannot get the pixel frames for the terminal state, try to fix it, although i don't think we can get any pixels for the terminal state
    # NOTE ::: Currently I am giving a small negative reward for sending commands, i.e for taking any action at all
    def GetMissionXML(self, mazeblock):
        return '''<?xml version="1.0" encoding="UTF-8" ?>
        <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
            <About>
                <Summary>Run the maze!</Summary>
            </About>

            <ModSettings>
                <MsPerTick>''' + str(self.TICK_LENGTH) + '''</MsPerTick>
            </ModSettings>

            <ServerSection>
                <ServerInitialConditions>
                    <Time>
                        <StartTime>1000</StartTime>
                        <AllowPassageOfTime>false</AllowPassageOfTime>
                    </Time>
                    <Weather>clear</Weather>
                    <AllowSpawning>false</AllowSpawning>
                </ServerInitialConditions>
                <ServerHandlers>
                    <FlatWorldGenerator generatorString="3;7,220*1,5*3,2;3;,biome_1" />
                    ''' + mazeblock + '''
                    <ServerQuitFromTimeUp timeLimitMs="8000"/>
                    <ServerQuitWhenAnyAgentFinishes />
                </ServerHandlers>
            </ServerSection>

            <AgentSection mode="Survival">
                <Name>LSD Curiosity</Name>
                <AgentStart>
                    <Placement x="-204" y="81" z="217"/>
                </AgentStart>
                <AgentHandlers>
                    <DiscreteMovementCommands/>
                    <VideoProducer want_depth="'''+str(self.want_depth_channel)+'''">
                        <Width>''' + str(self.video_width) + '''</Width>
                        <Height>''' + str(self.video_height) + '''</Height>
                    </VideoProducer>
                    <RewardForTouchingBlockType>
                        <Block reward="100.0" type="redstone_block" behaviour="onceOnly"/>
                        <Block reward="20.0" type="glowstone"/>
                        <Block reward="10.0" type="stone" variant="smooth_diorite"/>
                    </RewardForTouchingBlockType>
                    <RewardForSendingCommand reward="-1"/>
                </AgentHandlers>
            </AgentSection>

        </Mission>'''

    def take_action(self, action):
        if(self.world_state.is_mission_running):
            current_reward = 0
            while(not(self.world_state.number_of_observations_since_last_state > 0) and self.world_state.is_mission_running):
                time.sleep(0.05)
                self.world_state = self.agent_host.getWorldState()
            if(self.world_state.number_of_observations_since_last_state > 0 and self.world_state.is_mission_running):
                # print("Got " + str(self.world_state.number_of_observations_since_last_state) + " observations since last state.")
                try:
                    # for command in self.action_dict[action]:
                    #     self.agent_host.sendCommand(self.action_dict[action])
                    self.agent_host.sendCommand(self.action_dict[action])
                    # print('Took action ' + self.action_dict[action])
                    for reward in self.world_state.rewards:
                        current_reward += reward.getValue()
                    # if(current_reward>0):
                        # print('reward ::: '+str(current_reward))
                    while self.world_state.number_of_video_frames_since_last_state < 1 and self.world_state.is_mission_running:
                        time.sleep(0.05)
                        self.world_state = self.agent_host.getWorldState()
                    frame = self.world_state.video_frames[0]
                    frame_pixels = frame.pixels
                    next_state = np.frombuffer(frame_pixels, dtype=np.uint8)
                    is_terminal_flag = False
                    self.world_state = self.agent_host.getWorldState()
                    return (next_state, current_reward, is_terminal_flag)
                except RuntimeError as e:
                    print("Failed to send command:")
                    print(e)
                    return(None)
            else:
                if(not(self.world_state.is_mission_running)):
                    print("Mission has stopped.")
                    if (self.terminal_state_visit_count == 0):
                        time.sleep(0.5)  # Give mod a little time to get back to dormant state.
                        current_reward = 0
                        for reward in self.world_state.rewards:
                            current_reward += reward.getValue()
                        if (self.world_state.number_of_video_frames_since_last_state >= 1):
                            frame = self.world_state.video_frames[0]
                            frame_pixels = frame.pixels
                            next_state = np.frombuffer(frame_pixels, dtype=np.uint8)
                        else:
                            next_state = None
                        is_terminal_flag = True
                        self.terminal_state_visit_count += 1
                        self.world_state = self.agent_host.getWorldState()
                        return (next_state, current_reward, is_terminal_flag)
                    else:
                        # print('no obs for action ' + self.action_dict[action])
                        pass
                return(None)
        else:
            print("Mission has stopped.")
            if(self.terminal_state_visit_count==0):
                time.sleep(0.5)  # Give mod a little time to get back to dormant state.
                current_reward = 0
                for reward in self.world_state.rewards:
                    current_reward += reward.getValue()
                if(self.world_state.number_of_video_frames_since_last_state >= 1):
                    frame = self.world_state.video_frames[0]
                    frame_pixels = frame.pixels
                    next_state = np.frombuffer(frame_pixels, dtype=np.uint8)
                else:
                    next_state = None
                is_terminal_flag = True
                self.terminal_state_visit_count+=1
                self.world_state = self.agent_host.getWorldState()
                return (next_state, current_reward, is_terminal_flag)
            return(None)

    def get_current_state(self):
        self.world_state = self.agent_host.getWorldState()
        while(self.world_state.number_of_video_frames_since_last_state < 1 and self.world_state.is_mission_running):
            time.sleep(0.05)
            self.world_state = self.agent_host.getWorldState()

        if(self.world_state.number_of_video_frames_since_last_state >= 1):
            frame = self.world_state.video_frames[0]
            frame_pixels = frame.pixels
            current_state = np.frombuffer(frame_pixels, dtype=np.uint8)
            return current_state
        else:
            return(None)

def testing_function():
    maze_env = environment()
    maze_env.get_maze()
    total_num_actions_taken = 0
    frames_buffer = []
    for i in range(150):
        a = np.random.choice(list(maze_env.action_dict.keys()))
        action_result = maze_env.take_action(a)
        if(action_result):
            is_terminal = action_result[2]
            if(not(is_terminal)):
                s1 = action_result[0]
                frames_buffer.append(s1)
                r = action_result[1]
                total_num_actions_taken +=1
            else:
                print('Terminal state reached.....')
                s1 = action_result[0]
                frames_buffer.append(s1)
                r = action_result[1]
            print(s1)
            print(r)
            print(is_terminal)

    print('Total num actions taken is '+str(total_num_actions_taken))
    print('Length of total frames is '+str(len(frames_buffer)))

if(__name__=='__main__'):
    testing_function()

#
# #NOTE ::: The mission XML can also be generated programmatically as following:
# # my_mission = MalmoPython.MissionSpec()
# # my_mission.setSummary('A sample mission - run onto the gold block')
# # my_mission.requestVideo( 640, 480 )
# # my_mission.timeLimitInSeconds( 30 )
# # my_mission.allowAllChatCommands()
# # my_mission.allowAllInventoryCommands()
# # my_mission.setTimeOfDay( 1000, False )
# # my_mission.observeChat()
# # my_mission.observeGrid( -1, -1, -1, 1, 1, 1, 'grid' )
# # my_mission.observeHotBar()
# # my_mission.drawBlock( 5, 226, 5, 'gold_block' )
# # my_mission.rewardForReachingPosition( 5.5, 227, 5.5, 100, 0.5 )
# # my_mission.endAt( 5.5, 227, 5.5, 0.5 )
# # my_mission.startAt( 0.5, 227, 0.5 )
# # if rep % 2 == 1:  # alternate between continuous and discrete missions, for fun
# #     my_mission.removeAllCommandHandlers()
# #     my_mission.allowAllDiscreteMovementCommands()
# #
# # my_mission_record = MalmoPython.MissionRecordSpec('./hac_saved_mission_' + str(rep) + '.tgz')
# # my_mission_record.recordCommands()
# # my_mission_record.recordMP4(20, 400000)
# # my_mission_record.recordRewards()
# # my_mission_record.recordObservations()
