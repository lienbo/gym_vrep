# coding:utf-8

import os
import sys
import shutil
import subprocess
import numpy as np
import gym
from gym import spaces

######################################################
class VrepEnv:
    def __init__(self, scene="rollbalance", is_render=True, is_boot=True):
        # import V-REP
        if "linux" in sys.platform:
            self.VREP_DIR = os.path.expanduser("~") + "/V-REP_PRO_EDU/"
            vrepExe = "vrep.sh"
        elif "darwin" in sys.platform:
            self.VREP_DIR = "/Applications/V-REP_PRO_EDU/"
            vrepExe = "vrep.app/Contents/MacOS/vrep"
        else:
            print(sys.platform)
            sys.stderr.write("I don't know how to use vrep in Windows...\n")
            sys.exit(-1)
        sys.path.append(self.VREP_DIR + "programming/remoteApiBindings/python/python/")
        try:
            global vrep
            import vrep
        except:
            print ('--------------------------------------------------------------')
            print ('"vrep.py" could not be imported. This means very probably that')
            print ('either "vrep.py" or the remoteApi library could not be found.')
            print ('Make sure both are in the same folder as this file,')
            print ('or appropriately adjust the file "vrep.py"')
            print ('--------------------------------------------------------------')
            sys.exit(-1)
        # start V-REP
        self.IS_BOOT = is_boot
        if self.IS_BOOT:
            vrepArgs = [self.VREP_DIR + vrepExe, os.path.abspath(os.path.dirname(__file__))+"/scenes/"+scene+".ttt"]
            if not is_render:
                vrepArgs.extend(["-h"])
            vrepArgs.extend(["&"])
            self.vrepProcess = subprocess.Popen(vrepArgs, stdout=open(os.devnull, 'w'), stderr=subprocess.STDOUT, preexec_fn=os.setsid)
            print("Enviornment was opened: {}, {}".format(vrepArgs[0], vrepArgs[1]))
        # connect to V-REP
        ipAddress = "127.0.0.1"
        portNum = 19997
        self.__ID = vrep.simxStart(ipAddress, portNum, True, True, 5000, 1)
        while self.__ID == -1:
            self.__ID = vrep.simxStart(ipAddress, portNum, True, True, 5000, 1)
        # start to set constants
        vrep.simxSynchronous(self.__ID, True)
        vrep.simxStartSimulation(self.__ID, vrep.simx_opmode_blocking)
        vrep.simxSynchronousTrigger(self.__ID)
        if is_render:
            vrep.simxSetBooleanParameter(self.__ID, vrep.sim_boolparam_display_enabled, False, vrep.simx_opmode_oneshot)
        # constant shared with lua
        self.DT = vrep.simxGetFloatSignal(self.__ID, "dt", vrep.simx_opmode_blocking)[1]
        max_state = np.array(vrep.simxUnpackFloats(vrep.simxGetStringSignal(self.__ID, "max_state", vrep.simx_opmode_blocking)[1]))
        max_action = np.array(vrep.simxUnpackFloats(vrep.simxGetStringSignal(self.__ID, "max_action", vrep.simx_opmode_blocking)[1]))
        min_state = np.array(vrep.simxUnpackFloats(vrep.simxGetStringSignal(self.__ID, "min_state", vrep.simx_opmode_blocking)[1]))
        min_action = np.array(vrep.simxUnpackFloats(vrep.simxGetStringSignal(self.__ID, "min_action", vrep.simx_opmode_blocking)[1]))
        # limits of respective states and action
        self.observation_space = spaces.Box(min_state, max_state)
        self.action_space = spaces.Box(min_action, max_action)
        # variables will be received
        self.state = np.zeros(len(max_state))
        self.reward = 0.0
        self.done = False
        # variables will be sended
        self.action = np.zeros(len(max_action))
        # enable streaming
        self.state = np.array( vrep.simxUnpackFloats( vrep.simxGetStringSignal(self.__ID, "states", vrep.simx_opmode_streaming)[1] ) )
        self.reward = vrep.simxGetFloatSignal(self.__ID, "reward", vrep.simx_opmode_streaming)[1]
        self.done = bool( vrep.simxGetIntegerSignal(self.__ID, "done", vrep.simx_opmode_streaming)[1] )
        # stop simulation
        self.__stop()
        self.IS_RECORD = False



    def close(self):
        self.__stop()
        vrep.simxFinish(self.__ID)
        if self.IS_RECORD:
            self.__move()
        if self.IS_BOOT:
            import signal
            os.killpg(os.getpgid(self.vrepProcess.pid), signal.SIGTERM)
            self.vrepProcess.wait()
            print("Enviornment was closed")



    def reset(self):
        self.__stop()
        self.state = np.zeros_like(self.state)
        self.reward = 0.0
        self.done = False
        self.action = np.zeros_like(self.action)
        if self.IS_RECORD:
            self.__move()
            vrep.simxSetBooleanParameter(self.__ID, vrep.sim_boolparam_video_recording_triggered, True, vrep.simx_opmode_oneshot)
        vrep.simxSynchronous(self.__ID, True)
        vrep.simxStartSimulation(self.__ID, vrep.simx_opmode_blocking)
        self.__set(self.action)
        vrep.simxSynchronousTrigger(self.__ID)
        # get initial states
        self.__get()
        return self.state



    def step(self, action):
        # set actions
        self.__set(action)
        vrep.simxSynchronousTrigger(self.__ID)
        # get new states
        self.__get()
        return (self.state, self.reward, self.done, {})



    def monitor(self, save_dir="./video", force=False):
        self.IS_RECORD = True
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        if force:
            self.videoName = save_dir + "/recording"
        else:
            self.videoName = save_dir + "/"



    def __stop(self):
        vrep.simxSynchronous(self.__ID, False)
        vrep.simxStopSimulation(self.__ID, vrep.simx_opmode_blocking)
        while vrep.simxGetInMessageInfo(self.__ID, vrep.simx_headeroffset_server_state)[1] % 2 == 1:
            pass

    def __set(self, action):
        self.action = np.clip(action, self.action_space.low, self.action_space.high)
        vrep.simxSetStringSignal(self.__ID, "actions", vrep.simxPackFloats(self.action), vrep.simx_opmode_oneshot)

    def __get(self):
        vrep.simxGetPingTime(self.__ID)
        self.state = np.array( vrep.simxUnpackFloats( vrep.simxGetStringSignal(self.__ID, "states", vrep.simx_opmode_buffer)[1] ) )
        self.reward = vrep.simxGetFloatSignal(self.__ID, "reward", vrep.simx_opmode_buffer)[1]
        self.done = bool( vrep.simxGetIntegerSignal(self.__ID, "done", vrep.simx_opmode_buffer)[1] )

    def __move(self):
        for f in os.listdir(self.VREP_DIR):
            if "recording_" in f:
                if self.videoName[-1] == "/":
                    shutil.move(self.VREP_DIR + f, self.videoName + f)
                else:
                    name , ext = os.path.splitext(f)
                    shutil.move(self.VREP_DIR + f, self.videoName + ext)
