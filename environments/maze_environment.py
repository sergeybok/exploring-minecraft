import tensorflow as tf
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
                <SizeAndPosition length="60" width="60" yOrigin="225" zOrigin="0" height="180"/>
                <GapProbability variance="0.4">0.5</GapProbability>
                <Seed>random</Seed>
                <MaterialSeed>random</MaterialSeed>
                <AllowDiagonalMovement>false</AllowDiagonalMovement>
                <StartBlock fixedToEdge="true" type="emerald_block" height="1"/>
                <EndBlock fixedToEdge="true" type="redstone_block" height="12"/>
                <PathBlock type="stone dirt stained_hardened_clay" colour="WHITE ORANGE MAGENTA LIGHT_BLUE YELLOW LIME PINK GRAY SILVER CYAN PURPLE BLUE BROWN GREEN RED BLACK" height="1"/>
                <FloorBlock type="stone" variant="smooth_granite"/>
                <SubgoalBlock type="beacon sea_lantern glowstone"/>
                <OptimalPathBlock type="stone" variant="smooth_granite andesite smooth_diorite diorite"/>
                <GapBlock type="lapis_ore stained_hardened_clay air" colour="WHITE ORANGE MAGENTA LIGHT_BLUE YELLOW LIME PINK GRAY SILVER CYAN PURPLE BLUE BROWN GREEN RED BLACK" height="3" heightVariance="3"/>
                <AddQuitProducer description="finished maze"/>
                <AddNavigationObservations/>
            </MazeDecorator>
        '''

        self.video_width = 432
        self.video_height = 240
        self.validate = True

        self.agent_host = MalmoPython.AgentHost()
        self.agent_host.addOptionalIntArgument("speed,s", "Length of tick, in ms.", 50)
        try:
            self.agent_host.parse(sys.argv)
        except RuntimeError as e:
            print('ERROR:', e)
            print(self.agent_host.getUsage())
            exit(1)
        if self.agent_host.receivedArgument("help"):
            print(self.agent_host.getUsage())
            exit(0)

        self.agent_host.setObservationsPolicy(MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY)
        self.agent_host.setVideoPolicy(MalmoPython.VideoPolicy.LATEST_FRAME_ONLY)

        self.recordingsDirectory = "MazeRecordings"
        self.TICK_LENGTH = self.agent_host.getIntArgument("speed")

        try:
            os.makedirs(self.recordingsDirectory)
        except OSError as exception:
            if exception.errno != errno.EEXIST:  # ignore error if already existed
                raise


        self.mazeblocks = [self.maze4]
        self.curr_episode_num = 0


    def get_maze(self):
        # Set up a recording
        self.my_mission_record = MalmoPython.MissionRecordSpec()
        self.my_mission_record.recordRewards()
        self.my_mission_record.recordObservations()
        #TODO Figure out a way to increase the curr_episode_num and hence change the my_mission_record????
        self.my_mission_record.setDestination(self.recordingsDirectory + "//" + "Mission_" + str(self.curr_episode_num) + ".tgz")

        self.mazeblock = random.choice(self.mazeblocks)
        #TODO we should not update the 'my_mission' if we want the RL agent to play in the exact same maze again and again
        #TODO but we should update 'my_mission' if we want it to play the environment but different mazes
        #TODO Figure out if the maze is created afresh randomly by multiple calls to my_mission or it always creates the same maze???
        self.my_mission = MalmoPython.MissionSpec(self.GetMissionXML(self.mazeblock), self.validate)

        max_retries = 3
        for retry in range(max_retries):
            try:
                self.agent_host.startMission(self.my_mission, self.my_mission_record)
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:", e)
                    exit(1)
                else:
                    time.sleep(2)

        print("Waiting for the mission to start")
        self.world_state = self.agent_host.getWorldState()
        while not self.world_state.has_mission_begun:
            sys.stdout.write(".")
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            if len(world_state.errors):
                print()
                for error in world_state.errors:
                    print("Error:", error.text)
                    exit()
        print()

        self.curr_episode_num += 1


    def take_action(self, action):
        if(self.world_state.is_mission_running):
            if self.world_state.number_of_observations_since_last_state > 0:
                print("Got " + str(self.world_state.number_of_observations_since_last_state) + " observations since last state.")
                msg = self.world_state.observations[-1].text
                ob = json.loads(msg)
                #TODO understand the role of yawDelta and how is it used and it's relationship with current speed
                current_yaw_delta = ob.get(u'yawDelta', 0)
                current_speed = (1 - abs(current_yaw_delta))
                print("Got observation: " + str(current_yaw_delta))
                try:
                    #TODO figure out a way to convert the passed action into a command like "move 0.5" etc.
                    #TODO i can use a dictionary for this like {0: 'move 0.5', 1: 'turn 0.5'} etc.
                    self.agent_host.sendCommand(action)
                    # self.agent_host.sendCommand("move " + str(current_speed))
                    # self.agent_host.sendCommand("turn " + str(current_yaw_delta))
                except RuntimeError as e:
                    print("Failed to send command:", e)
                    pass
            self.world_state = self.agent_host.getWorldState()
            #TODO define the value of next_state and next_reward
            frame = self.world_state.video_frames[0]
            image_pixels = np.frombuffer(frame.pixels, dtype=np.uint8)
            image_pixels = image_pixels.reshape((frame.height, frame.width, 4))
            #TODO Figure out whether to use the depth channel or not
            image_pixels = image_pixels[:, :, 0:3]
            #TODO figure out whether to pass the flattened image or the reshaped image to CNN
            image_pixels = image_pixels.flatten()
            next_state = image_pixels

            current_reward = 0
            for reward in self.world_state.rewards:
                current_reward += reward.getValue()
            is_terminal_flag = False
            return (next_state, current_reward, is_terminal_flag)
        else:
            print("Mission has stopped.")
            time.sleep(0.5)  # Give mod a little time to get back to dormant state.
            # TODO Figure out whether the below definitions of state and reward work at the terminal state or not
            # TODO if the below definitions work at terminal state as well then combine the above if and this else conditions and
            # TODO change only the is_Terminal_flag
            self.world_state = self.agent_host.getWorldState()
            # TODO define the value of next_state and next_reward
            frame = self.world_state.video_frames[0]
            image_pixels = np.frombuffer(frame.pixels, dtype=np.uint8)
            image_pixels = image_pixels.reshape((frame.height, frame.width, 4))
            # TODO Figure out whether to use the depth channel or not
            image_pixels = image_pixels[:, :, 0:3]
            # TODO figure out whether to pass the flattened image or the reshaped image to CNN
            image_pixels = image_pixels.flatten()
            next_state = image_pixels
            current_reward = 0
            for reward in self.world_state.rewards:
                current_reward += reward.getValue()
            is_terminal_flag = True
            return (next_state, current_reward, is_terminal_flag)

    def get_current_state(self):
        #TODO define current_state
        self.world_state = self.agent_host.getWorldState()
        # TODO define the value of next_state and next_reward
        frame = self.world_state.video_frames[0]
        image_pixels = np.frombuffer(frame.pixels, dtype=np.uint8)
        image_pixels = image_pixels.reshape((frame.height, frame.width, 4))
        # TODO Figure out whether to use the depth channel or not
        image_pixels = image_pixels[:, :, 0:3]
        # TODO figure out whether to pass the flattened image or the reshaped image to CNN
        image_pixels = image_pixels.flatten()
        current_state = image_pixels
        return current_state

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
                    <VideoProducer want_depth="true">
                            <Width>''' + str(self.video_width) + '''</Width>
                            <Height>''' + str(self.video_height) + '''</Height>
                    </VideoProducer>
                    <ContinuousMovementCommands turnSpeedDegs="840">
                        <ModifierList type="deny-list"> <!-- Example deny-list: prevent agent from strafing -->
                            <command>strafe</command>
                        </ModifierList>
                    </ContinuousMovementCommands>
                </AgentHandlers>
            </AgentSection>

        </Mission>'''
