#   Distributed W-Learning based Spatio-Temporal Variable Speed Limit (DWL-ST-VSL)
#   Author: Krešimir Kušić

#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.

#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.

#   You should have received a copy of the GNU General Public License
#   along with this program. If not, see <http://www.gnu.org/licenses/>


import os, sys
import traci # Traffic Control Interface
import traci.constants as tc
import shlex, subprocess
import random
import numpy as np
import matplotlib.pyplot as plt
import itertools
from numpy import savetxt
from xml.dom import minidom as mini
import time
import math
import datetime
import pandas as pd
from copy import deepcopy




# Defintion of SUMO path, root of the model and files , path to save results 
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
        
sumoBinary = "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo-gui" #remove (-gui) if you whant to speed up sim. process
root="C:/Users/kusic/Desktop/DWL-ST-VSL/model_MD/"
path_results = "C:/Users/kusic/Desktop/DWL-ST-VSL/Results/"
sumoCmd = [sumoBinary, "-c",\
           root+"highway_model.sumocfg", "--seed", str(28815),\
                "--start", "1", "--quit-on-end", "1"]






# Definition of "Agents", functions...
class IterAgents(type):
    def __iter__(cls):
        return iter(cls._allAgents)
    
class Agent(metaclass=IterAgents):
    _allAgents=[]

    def __init__(self, name):
        
        self._allAgents.append(self)
        self.name = name
    # Q-Learning update rule   
    def updateQ(self, x, ak, y, action_subset_y, r, alpha_Q_ak, gama, Q):
        Q[x,ak] = (1 - alpha_Q_ak)*Q[x, ak] + alpha_Q_ak*(r + gama*(Q[y,action_subset_y].max()))
    # W-Learning update rule   
    def updateW(self, x, ai, y, action_subset_y, r, alpha_W, alpha_Q_ai, gama, w, W, Q):
        W[x] = (1 - alpha_W)*W[x] + alpha_W*((1-alpha_Q_ai)**w)*(Q[x,ai] - \
                (r + gama*(Q[y,action_subset_y].max())))
        
    def alpha_Q_ak_Function(self, x, ak, Num_Visited_x_ak, update):
        if update == 1:
            Num_Visited_x_ak[x,ak]+=1
            alpha_Q_ak = 1/Num_Visited_x_ak[x,ak]
        else:
            if Num_Visited_x_ak[x,ak] == 0:
                alpha_Q_ak = 1
            else:
                alpha_Q_ak = 1/Num_Visited_x_ak[x,ak]
        return alpha_Q_ak
    
    def alpha_W_Function(self, x, Num_Visited_x, update):
        if update == 1:
            Num_Visited_x[x,0]+=1  
            alpha_W = 1/Num_Visited_x[x,0] 
        else:
            if Num_Visited_x[x,0] == 0:
                    alpha_W = 1
            else:
                alpha_W = 1/Num_Visited_x[x,0]
        return alpha_W
        
    def alpha_Q_ai_Function(self, x, ai, Num_Visited_x_ai):
        Num_Visited_x_ai[x,ai]+=1
        alpha_Q_ai = 1/Num_Visited_x_ai[x,ai]
        return alpha_Q_ai
    
        
    def suggestAction(self, epsilon, x, Q, previous_action_index, actions_A, run, numberSaveData):
        random_number = random.random()
        x_random = sum(random_number >= np.cumsum([0,1-epsilon,epsilon]))
            
        x1=np.array(abs(actions_A[:] - actions_A[previous_action_index])<=20)
        x2=np.array(abs(actions_A[:] - actions_A[previous_action_index])<=20)

        intersect_logic = np.logical_and(x1, x2)
        intersect = np.where(intersect_logic == True)
        action_subset = intersect[0]
        
        if x_random == 1: # exploit
            index_max_Q = np.random.choice(np.flatnonzero(Q[x,action_subset]\
                                                           == Q[x,action_subset].max()))
            action_index=action_subset[index_max_Q]
            action_control = 0
            
        else:  # explore
            action_index = np.random.choice(action_subset)
            action_control = 1
            
        return action_index, action_control
    
    
    
    def subsetAction_in_y(self, actionWin, actions_A, run, numberSaveData):
        x1=np.array(abs(actions_A[:] - actions_A[actionWin])<=20)
        x2=np.array(abs(actions_A[:] - actions_A[actionWin])<=20)
        intersect_logic = np.logical_and(x1, x2)
        intersect = np.where(intersect_logic == True)
        action_subset_y = intersect[0]
        
        return action_subset_y

# ======= Distributed W-Learning (DWL)
# agent
# C - cooperation coefficient
# x_LPi - Local Policy state
# x_RPi - Remote Policy state 
# W_LPi - W matrix of LP policy
# W_RPi - W matrix of RP policy

def actionWinner(agent, C, x_LP1, x_LP2, x_RP1, x_RP2, epsilon, actionIndexLP1, actionIndexLP2,\
                 actionIndexRP1, actionIndexRP2, W_LP1, W_LP2, W_RP1, W_RP2):
    
    action_index_LP = [actionIndexLP1, actionIndexLP2]
    action_index_RP = [actionIndexRP1, actionIndexRP2]
    Ws_LP = np.array([[W_LP1[x_LP1,0], W_LP2[x_LP2,0]]])

    index_max_W_LP = np.random.choice(np.flatnonzero(Ws_LP[0,:] == Ws_LP[0,:].max())) # to indicate policy "LPi"
    action_win_index_LP = action_index_LP[index_max_W_LP]
    agentAiLPi_win = "LP"+str(index_max_W_LP+1)
    AiLP1want = actionIndexLP1
    AiLP2want = actionIndexLP2

    Ws_RP = np.array([[W_RP1[x_RP1,0], W_RP2[x_RP2,0]]])
    index_max_W_RP = np.random.choice(np.flatnonzero(Ws_RP[0,:] == Ws_RP[0,:].max())) # to indicate policy "LPi"
    action_win_index_RP = action_index_RP[index_max_W_RP]
    agentAiRPj_win = "RP"+str(index_max_W_RP+1)
    AiRP1want = actionIndexRP1
    AiRP2want = actionIndexRP2
    
    W_win = np.array([[Ws_LP[0, index_max_W_LP], C*Ws_RP[0, index_max_W_RP]]])
    
    index_max_W_win = np.random.choice(np.flatnonzero(W_win[0,:] == W_win[0,:].max())) # to indicate policy "LPi"

    if index_max_W_win == 0:
        policyWin = "LP"+str(index_max_W_LP+1)
        action_win_index = action_win_index_LP
    else:
        policyWin = "RP"+str(index_max_W_RP+1)
        action_win_index = action_win_index_RP
    
    return AiLP1want, AiLP2want, agentAiLPi_win, AiRP1want, AiRP2want, agentAiRPj_win, action_win_index, policyWin
    

# Initialization of agents from Class "Agent"    
A1 = Agent('A1')
A2 = Agent('A2')

# Q and W matrices associated with Agents' Local and Remote policies
A1.Q_LPi=[np.zeros((4608,8)), np.zeros((4608,8))]  # 4608 states, 8 actions
A2.Q_LPi=[np.zeros((4608,8)), np.zeros((4608,8))]
A1.Q_RPi=[np.zeros((4608,8)), np.zeros((4608,8))]
A2.Q_RPi=[np.zeros((4608,8)), np.zeros((4608,8))]
A1.W_LPi=[np.zeros((4608,1)), np.zeros((4608,1))]
A2.W_LPi=[np.zeros((4608,1)), np.zeros((4608,1))]
A1.W_RPi=[np.zeros((4608,1)), np.zeros((4608,1))]
A2.W_RPi=[np.zeros((4608,1)), np.zeros((4608,1))]

# Tracking the number of visited states 
A1.Num_Visited_LPi_x_ak=[np.zeros((4608,8)), np.zeros((4608,8))]
A2.Num_Visited_LPi_x_ak=[np.zeros((4608,8)), np.zeros((4608,8))]
A1.Num_Visited_RPi_x_ak=[np.zeros((4608,8)), np.zeros((4608,8))]
A2.Num_Visited_RPi_x_ak=[np.zeros((4608,8)), np.zeros((4608,8))]
A1.Num_Visited_LPi_x_ai=[np.zeros((4608,1)), np.zeros((4608,1))]
A2.Num_Visited_LPi_x_ai=[np.zeros((4608,1)), np.zeros((4608,1))]
A1.Num_Visited_RPi_x_ai=[np.zeros((4608,1)), np.zeros((4608,1))]
A2.Num_Visited_RPi_x_ai=[np.zeros((4608,1)), np.zeros((4608,1))]
A1.Num_Visited_LPi_x=[np.zeros((4608,1)), np.zeros((4608,1))]
A2.Num_Visited_LPi_x=[np.zeros((4608,1)), np.zeros((4608,1))]
A1.Num_Visited_RPi_x=[np.zeros((4608,1)), np.zeros((4608,1))]
A2.Num_Visited_RPi_x=[np.zeros((4608,1)), np.zeros((4608,1))]



# State definition (for details see our papers mentioned in README file)
states_combinationsAi = np.empty([4608, 4], dtype=object)
k = 0
for i in itertools.product(['a0','a1','a2','a3','a4','a5','a6','a7'],\
                           ['A1v1','A1v2','A1v3','A1v4'],\
                           ['L3p1','L3p2','L3p3','L3p4','L3p5','L3p6','L3p7','L3p8','L3p9','L3p10','L3p11','L3p12'],\
                           ['L4p1','L4p2','L4p3','L4p4','L4p5','L4p6','L4p7','L4p8','L4p9','L4p10','L4p11','L4p12']):

    states_combinationsAi[k, 0]=i[0]
    states_combinationsAi[k, 1]=i[1]
    states_combinationsAi[k, 2]=i[2]
    states_combinationsAi[k, 3]=i[3]
    k+=1
    
v1=14
v2=21
v3=28



p1=15
p2=20
p3=23
p4=26
p5=29
p6=32
p7=35
p8=38
p9=45
p10=55
p11=65


# It's not actually mv "momentum", rather it is an association that we use speed and 
# number of vehicles (density -> mass) for states in this case, together with action from previous control step
def returnStateMomentum(a_before, v_in, p_in1, p_in2):
    
    if a_before==0:
        a_ = 'a0'
    elif a_before==1:
        a_ = 'a1'
    elif a_before==2:
        a_ = 'a2'
    elif a_before==3:
        a_ = 'a3'
    elif a_before==4:
        a_ = 'a4'
    elif a_before==5:
        a_ = 'a5'
    elif a_before==6:
        a_ = 'a6'
    else:
        a_ = 'a7'
        
  
    if v_in<=v1:
        A1v = 'A1v1'
    elif v1<v_in<=v2:
        A1v = 'A1v2'
    elif v2<v_in<=v3:
        A1v = 'A1v3'
    else:
        A1v = 'A1v4'
              
          
    if p_in1<p1:
        L3p = 'L3p1'
    elif p1<=p_in1<p2:
        L3p = 'L3p2'
    elif p2<=p_in1<p3:
        L3p = 'L3p3'
    elif p3<=p_in1<p4:
        L3p = 'L3p4'
    elif p4<=p_in1<p5:
        L3p = 'L3p5'
    elif p5<=p_in1<p6:
        L3p = 'L3p6'
    elif p6<=p_in1<p7:
        L3p = 'L3p7'
    elif p7<=p_in1<p8:
        L3p = 'L3p8'
    elif p8<=p_in1<p9:
        L3p = 'L3p9'  
    elif p9<=p_in1<p10:
        L3p = 'L3p10'  
    elif p10<=p_in1<p11:
        L3p = 'L3p11'  
    else:
        L3p = 'L3p12'
        
        
    if p_in2<p1:
        L4p = 'L4p1'
    elif p1<=p_in2<p2:
        L4p = 'L4p2'
    elif p2<=p_in2<p3:
        L4p = 'L4p3'
    elif p3<=p_in2<p4:
        L4p = 'L4p4'
    elif p4<=p_in2<p5:
        L4p = 'L4p5'
    elif p5<=p_in2<p6:
        L4p = 'L4p6'
    elif p6<=p_in2<p7:
        L4p = 'L4p7'
    elif p7<=p_in2<p8:
        L4p = 'L4p8'
    elif p8<=p_in2<p9:
        L4p = 'L4p9'
    elif p9<=p_in2<p10:
        L4p = 'L4p10'
    elif p10<=p_in2<p11:
        L4p = 'L4p11'
    else:
        L4p = 'L4p12'
    
    
    s1 = np.where(states_combinationsAi[:,0] == a_)
    s2 = np.where(states_combinationsAi[:,1] == A1v)
    s3 = np.where(states_combinationsAi[:,2] == L3p)
    s4 = np.where(states_combinationsAi[:,3] == L4p)

    s1_set = set(np.unique(s1))
    s2_set = set(np.unique(s2))
    s3_set = set(np.unique(s3))
    s4_set = set(np.unique(s4))
    row_intersect = set(s1_set).intersection(s2_set, s3_set, s4_set)
    list_row_intersect = list(row_intersect)
    row_state_indexA1 = list_row_intersect[0]

    return row_state_indexA1
    
    
    
    
    
def saveData(vsl_on_off, run, numberSaveData, end_learning):
    if vsl_on_off==1 and (run%numberSaveData==0 or run>end_learning):
        
        savetxt(path_results+"speedHeatMap"+str(run)+".csv", speedHeatMap, delimiter=',')
        savetxt(path_results+"speedLimitHeatMap"+str(run)+".csv", speedLimitHeatMap, delimiter=',')        
        
        # save A1 policies LPs and RPs
        savetxt(path_results+"A1LP1"+str(run)+".csv", A1.Q_LPi[0], delimiter=',')
        savetxt(path_results+"A1LP2"+str(run)+".csv", A1.Q_LPi[1], delimiter=',')
        savetxt(path_results+"A1RP1"+str(run)+".csv", A1.Q_RPi[0], delimiter=',')
        savetxt(path_results+"A1RP2"+str(run)+".csv", A1.Q_RPi[1], delimiter=',')
        # save A2 policies LPs and RPs
        savetxt(path_results+"A2LP1"+str(run)+".csv", A2.Q_LPi[0], delimiter=',')
        savetxt(path_results+"A2LP2"+str(run)+".csv", A2.Q_LPi[1], delimiter=',')
        savetxt(path_results+"A2RP1"+str(run)+".csv", A2.Q_RPi[0], delimiter=',')
        savetxt(path_results+"A2RP2"+str(run)+".csv", A2.Q_RPi[1], delimiter=',')
        # save A1 Ws
        savetxt(path_results+"A1W_LP1"+str(run)+".csv", A1.W_LPi[0], delimiter=',')
        savetxt(path_results+"A1W_LP2"+str(run)+".csv", A1.W_LPi[1], delimiter=',')
        savetxt(path_results+"A1W_RP1"+str(run)+".csv", A1.W_RPi[0], delimiter=',')
        savetxt(path_results+"A1W_RP2"+str(run)+".csv", A1.W_RPi[1], delimiter=',')
        # save A2 Ws
        savetxt(path_results+"A2W_LP1"+str(run)+".csv", A2.W_LPi[0], delimiter=',')
        savetxt(path_results+"A2W_LP2"+str(run)+".csv", A2.W_LPi[1], delimiter=',')
        savetxt(path_results+"A2W_RP1"+str(run)+".csv", A2.W_RPi[0], delimiter=',')
        savetxt(path_results+"A2W_RP2"+str(run)+".csv", A2.W_RPi[1], delimiter=',')

        # save n(x,ak) for A1, A2
        savetxt(path_results+"A1NumLP1_X_ak"+str(run)+".csv", A1.Num_Visited_LPi_x_ak[0], delimiter=',')
        savetxt(path_results+"A1NumLP2_X_ak"+str(run)+".csv", A1.Num_Visited_LPi_x_ak[1], delimiter=',')
        savetxt(path_results+"A1NumRP1_X_ak"+str(run)+".csv", A1.Num_Visited_RPi_x_ak[0], delimiter=',')
        savetxt(path_results+"A1NumRP2_X_ak"+str(run)+".csv", A1.Num_Visited_RPi_x_ak[1], delimiter=',')
        
        savetxt(path_results+"A2NumLP1_X_ak"+str(run)+".csv", A2.Num_Visited_LPi_x_ak[0], delimiter=',')
        savetxt(path_results+"A2NumLP2_X_ak"+str(run)+".csv", A2.Num_Visited_LPi_x_ak[1], delimiter=',')
        savetxt(path_results+"A2NumRP1_X_ak"+str(run)+".csv", A2.Num_Visited_RPi_x_ak[0], delimiter=',')
        savetxt(path_results+"A2NumRP2_X_ak"+str(run)+".csv", A2.Num_Visited_RPi_x_ak[1], delimiter=',')
        
        # save n(x,ai) for A1, A2 this is not necessary
        savetxt(path_results+"A1NumLP1_X_ai"+str(run)+".csv", A1.Num_Visited_LPi_x_ai[0], delimiter=',')
        savetxt(path_results+"A1NumLP2_X_ai"+str(run)+".csv", A1.Num_Visited_LPi_x_ai[1], delimiter=',')
        savetxt(path_results+"A1NumRP1_X_ai"+str(run)+".csv", A1.Num_Visited_RPi_x_ai[0], delimiter=',')
        savetxt(path_results+"A1NumRP2_X_ai"+str(run)+".csv", A1.Num_Visited_RPi_x_ai[1], delimiter=',')
        
        savetxt(path_results+"A2NumLP1_X_ai"+str(run)+".csv", A2.Num_Visited_LPi_x_ai[0], delimiter=',')
        savetxt(path_results+"A2NumLP2_X_ai"+str(run)+".csv", A2.Num_Visited_LPi_x_ai[1], delimiter=',')
        savetxt(path_results+"A2NumRP1_X_ai"+str(run)+".csv", A2.Num_Visited_RPi_x_ai[0], delimiter=',')
        savetxt(path_results+"A2NumRP2_X_ai"+str(run)+".csv", A2.Num_Visited_RPi_x_ai[1], delimiter=',')
        
        # save n(x) for A1, A2
        savetxt(path_results+"A1NumLP1_X"+str(run)+".csv", A1.Num_Visited_LPi_x[0], delimiter=',')
        savetxt(path_results+"A1NumLP2_X"+str(run)+".csv", A1.Num_Visited_LPi_x[1], delimiter=',')
        savetxt(path_results+"A1NumRP1_X"+str(run)+".csv", A1.Num_Visited_RPi_x[0], delimiter=',')
        savetxt(path_results+"A1NumRP2_X"+str(run)+".csv", A1.Num_Visited_RPi_x[1], delimiter=',')
        
        savetxt(path_results+"A2NumLP1_X"+str(run)+".csv", A2.Num_Visited_LPi_x[0], delimiter=',')
        savetxt(path_results+"A2NumLP2_X"+str(run)+".csv", A2.Num_Visited_LPi_x[1], delimiter=',')
        savetxt(path_results+"A2NumRP1_X"+str(run)+".csv", A2.Num_Visited_RPi_x[0], delimiter=',')
        savetxt(path_results+"A2NumRP2_X"+str(run)+".csv", A2.Num_Visited_RPi_x[1], delimiter=',')
               
        # optional -> Results and additional analytics for control of the agent's learning process (you can reduce it unnecessary)
        df=pd.DataFrame({'A1X(t-1)': controlFile[1:,0], 'A2X(t-1)': controlFile[1:,1], 'V1(t-1)': controlFile[1:,2],\
                        'V2(t-1)': controlFile[1:,3],'V3(t-1)': controlFile[1:,4],'V4(t-1)': controlFile[1:,5],\
                        'denL1(t-1)': controlFile[1:,6],'denL2(t-1)': controlFile[1:,7], 'denL3(t-1)': controlFile[1:,8],\
                        'denL4(t-1)': controlFile[1:,9], 'VSL1(t-1)': controlFile[1:,10], 'VSL2(t-1)': controlFile[1:,11],\
                        'TTS1(t-1)': controlFile[1:,12], 'TTS2(t-1)': controlFile[1:,13], 'TTS3(t-1)': controlFile[1:,14],\
                        'TTS4(t-1)': controlFile[1:,15], 'TTSW(t-1)': controlFile[1:,16], 'NumVeh_W': controlFile[1:,17],\
                        'A1r1': controlFile[1:,18], 'A1r2': controlFile[1:,19], 'A2r1': controlFile[1:,20],\
                        'A2r2': controlFile[1:,21], 'A1X(t)': controlFile[1:,22], 'A2X(t)': controlFile[1:,23],\
                        'V1(t)': controlFile[1:,24],'V2(t)': controlFile[1:,25],'V3(t)': controlFile[1:,26],\
                        'V4(t)': controlFile[1:,27], 'denL1(t)': controlFile[1:,28],\
                         'denL2(t)': controlFile[1:,29], 'denL3(t)': controlFile[1:,30],\
                         'denL4(t)': controlFile[1:,31], 'VSL1(t)': controlFile[1:,32],\
                        'A1LP1want': controlFile[1:,33], 'A1LP2want': controlFile[1:,34],'A1RP1want': controlFile[1:,35],\
                        'A1RP2want': controlFile[1:,36], 'A1policyWin': controlFile[1:,37],'A1LP1cont': controlFile[1:,38],\
                        'A1LP2cont': controlFile[1:,39], 'A1RP1cont': controlFile[1:,40],'A1RP2cont': controlFile[1:,41],\
                        'VSL2(t)': controlFile[1:,42], 'A2LP1want': controlFile[1:,43],'A2LP2want': controlFile[1:,44],\
                         'A2RP1want': controlFile[1:,45], 'A2RP2want': controlFile[1:,46],'A2policyWin': controlFile[1:,47],\
                         'A2LP1cont': controlFile[1:,48], 'A2LP2cont': controlFile[1:,49], 'A2RP1cont': controlFile[1:,50],\
                         'A2RP2cont': controlFile[1:,51], 'A1W_LP1': controlFile[1:,52],'A1W_LP2': controlFile[1:,53],\
                        'A1W_RP1': controlFile[1:,54],'A1W_RP2': controlFile[1:,55],'A2W_LP1': controlFile[1:,56],\
                         'A2W_LP2': controlFile[1:,57], 'A2W_RP1': controlFile[1:,58],'A2W_RP2': controlFile[1:,59],\
                        'A1alph_LP1_ai': controlFile[1:,60],'A1alph_LP2_ai': controlFile[1:,61],\
                        'A1alph_RP1_ai': controlFile[1:,62],'A1alph_RP2_ai': controlFile[1:,63],\
                         'A2alph_LP1_ai': controlFile[1:,64],'A2alph_LP2_ai': controlFile[1:,65],\
                         'A2alph_RP1_ai': controlFile[1:,66],'A2alph_RP2_ai': controlFile[1:,67],\
                        'A1alph_LP1_ak': controlFile[1:,68],'A1alph_LP2_ak': controlFile[1:,69],\
                        'A1alph_RP1_ak': controlFile[1:,70],'A1alph_RP2_ak': controlFile[1:,71],\
                         'A2alph_LP1_ak': controlFile[1:,72],'A2alph_LP2_ak': controlFile[1:,73],\
                         'A2alph_RP1_ak': controlFile[1:,74],'A2alph_RP2_ak': controlFile[1:,75],\
                        'A1alph_WLP1': controlFile[1:,76],'A1alph_WLP2': controlFile[1:,77],\
                        'A1alph_WRP1': controlFile[1:,78],'A1alph_WRP2': controlFile[1:,79],\
                         'A2alph_WLP1': controlFile[1:,80],'A2alph_WLP2': controlFile[1:,81],\
                         'A2alph_WRP1': controlFile[1:,82],'A2alph_WRP2': controlFile[1:,83],\
                        'A1alt_WLP1': controlFile[1:,84],'A1alt_WLP2': controlFile[1:,85],\
                        'A1alt_WRP1': controlFile[1:,86],'A1alt_WRP2': controlFile[1:,87],\
                         'A2alt_WLP1': controlFile[1:,88],'A2alt_WLP2': controlFile[1:,89],\
                         'A2alt_WRP1': controlFile[1:,90],'A2alt_WRP2': controlFile[1:,91],\
                         'TTSsim': controlFile[1:,92], 'Occ1': controlFile[1:,93], 'Occ2': controlFile[1:,94],\
                        'Occ3': controlFile[1:,95], 'A1win': controlFile[1:,96], 'A1LPwin': controlFile[1:,97],\
                        'A1RPwin': controlFile[1:,98], 'A2win': controlFile[1:,99], 'A2LPwin': controlFile[1:,100],\
                        'A2RPwin': controlFile[1:,101], 'VSL1pos': controlFile[1:,102], 'VSL2pos': controlFile[1:,103]})
        
        df.to_excel(path_results+"controlFile"+str(run)+".xlsx", index = False)

        
    elif vsl_on_off==0:
        
        savetxt(path_results+"NO_VSL_speedHeatMap"+str(run)+".csv", speedHeatMap, delimiter=',')
        savetxt(path_results+"NO_VSL_speedLimitHeatMap"+str(run)+".csv", speedLimitHeatMap, delimiter=',')        
        
        # save A1 policies LPs and RPs
        savetxt(path_results+"NO_VSL_A1LP1"+str(run)+".csv", A1.Q_LPi[0], delimiter=',')
        savetxt(path_results+"NO_VSL_A1LP2"+str(run)+".csv", A1.Q_LPi[1], delimiter=',')
        savetxt(path_results+"NO_VSL_A1RP1"+str(run)+".csv", A1.Q_RPi[0], delimiter=',')
        savetxt(path_results+"NO_VSL_A1RP2"+str(run)+".csv", A1.Q_RPi[1], delimiter=',')
        # save A2 policies LPs and RPs
        savetxt(path_results+"NO_VSL_A2LP1"+str(run)+".csv", A2.Q_LPi[0], delimiter=',')
        savetxt(path_results+"NO_VSL_A2LP2"+str(run)+".csv", A2.Q_LPi[1], delimiter=',')
        savetxt(path_results+"NO_VSL_A2RP1"+str(run)+".csv", A2.Q_RPi[0], delimiter=',')
        savetxt(path_results+"NO_VSL_A2RP2"+str(run)+".csv", A2.Q_RPi[1], delimiter=',')
        # save A1 Ws
        savetxt(path_results+"NO_VSL_A1W_LP1"+str(run)+".csv", A1.W_LPi[0], delimiter=',')
        savetxt(path_results+"NO_VSL_A1W_LP2"+str(run)+".csv", A1.W_LPi[1], delimiter=',')
        savetxt(path_results+"NO_VSL_A1W_RP1"+str(run)+".csv", A1.W_RPi[0], delimiter=',')
        savetxt(path_results+"NO_VSL_A1W_RP2"+str(run)+".csv", A1.W_RPi[1], delimiter=',')
        # save A2 Ws
        savetxt(path_results+"NO_VSL_A2W_LP1"+str(run)+".csv", A2.W_LPi[0], delimiter=',')
        savetxt(path_results+"NO_VSL_A2W_LP2"+str(run)+".csv", A2.W_LPi[1], delimiter=',')
        savetxt(path_results+"NO_VSL_A2W_RP1"+str(run)+".csv", A2.W_RPi[0], delimiter=',')
        savetxt(path_results+"NO_VSL_A2W_RP2"+str(run)+".csv", A2.W_RPi[1], delimiter=',')

        # save n(x,ak) for A1, A2
        savetxt(path_results+"NO_VSL_A1NumLP1_X_ak"+str(run)+".csv", A1.Num_Visited_LPi_x_ak[0], delimiter=',')
        savetxt(path_results+"NO_VSL_A1NumLP2_X_ak"+str(run)+".csv", A1.Num_Visited_LPi_x_ak[1], delimiter=',')
        savetxt(path_results+"NO_VSL_A1NumRP1_X_ak"+str(run)+".csv", A1.Num_Visited_RPi_x_ak[0], delimiter=',')
        savetxt(path_results+"NO_VSL_A1NumRP2_X_ak"+str(run)+".csv", A1.Num_Visited_RPi_x_ak[1], delimiter=',')
        
        savetxt(path_results+"NO_VSL_A2NumLP1_X_ak"+str(run)+".csv", A2.Num_Visited_LPi_x_ak[0], delimiter=',')
        savetxt(path_results+"NO_VSL_A2NumLP2_X_ak"+str(run)+".csv", A2.Num_Visited_LPi_x_ak[1], delimiter=',')
        savetxt(path_results+"NO_VSL_A2NumRP1_X_ak"+str(run)+".csv", A2.Num_Visited_RPi_x_ak[0], delimiter=',')
        savetxt(path_results+"NO_VSL_A2NumRP2_X_ak"+str(run)+".csv", A2.Num_Visited_RPi_x_ak[1], delimiter=',')
        
        # save n(x,ai) for A1, A2 this is not necessary
        savetxt(path_results+"NO_VSL_A1NumLP1_X_ai"+str(run)+".csv", A1.Num_Visited_LPi_x_ai[0], delimiter=',')
        savetxt(path_results+"NO_VSL_A1NumLP2_X_ai"+str(run)+".csv", A1.Num_Visited_LPi_x_ai[1], delimiter=',')
        savetxt(path_results+"NO_VSL_A1NumRP1_X_ai"+str(run)+".csv", A1.Num_Visited_RPi_x_ai[0], delimiter=',')
        savetxt(path_results+"NO_VSL_A1NumRP2_X_ai"+str(run)+".csv", A1.Num_Visited_RPi_x_ai[1], delimiter=',')
        
        savetxt(path_results+"NO_VSL_A2NumLP1_X_ai"+str(run)+".csv", A2.Num_Visited_LPi_x_ai[0], delimiter=',')
        savetxt(path_results+"NO_VSL_A2NumLP2_X_ai"+str(run)+".csv", A2.Num_Visited_LPi_x_ai[1], delimiter=',')
        savetxt(path_results+"NO_VSL_A2NumRP1_X_ai"+str(run)+".csv", A2.Num_Visited_RPi_x_ai[0], delimiter=',')
        savetxt(path_results+"NO_VSL_A2NumRP2_X_ai"+str(run)+".csv", A2.Num_Visited_RPi_x_ai[1], delimiter=',')
        
        # save n(x) for A1, A2
        savetxt(path_results+"NO_VSL_A1NumLP1_X"+str(run)+".csv", A1.Num_Visited_LPi_x[0], delimiter=',')
        savetxt(path_results+"NO_VSL_A1NumLP2_X"+str(run)+".csv", A1.Num_Visited_LPi_x[1], delimiter=',')
        savetxt(path_results+"NO_VSL_A1NumRP1_X"+str(run)+".csv", A1.Num_Visited_RPi_x[0], delimiter=',')
        savetxt(path_results+"NO_VSL_A1NumRP2_X"+str(run)+".csv", A1.Num_Visited_RPi_x[1], delimiter=',')
        
        savetxt(path_results+"NO_VSL_A2NumLP1_X"+str(run)+".csv", A2.Num_Visited_LPi_x[0], delimiter=',')
        savetxt(path_results+"NO_VSL_A2NumLP2_X"+str(run)+".csv", A2.Num_Visited_LPi_x[1], delimiter=',')
        savetxt(path_results+"NO_VSL_A2NumRP1_X"+str(run)+".csv", A2.Num_Visited_RPi_x[0], delimiter=',')
        savetxt(path_results+"NO_VSL_A2NumRP2_X"+str(run)+".csv", A2.Num_Visited_RPi_x[1], delimiter=',')
        
        df=pd.DataFrame({'A1X(t-1)': controlFile[1:,0], 'A2X(t-1)': controlFile[1:,1], 'V1(t-1)': controlFile[1:,2],\
                        'V2(t-1)': controlFile[1:,3],'V3(t-1)': controlFile[1:,4],'V4(t-1)': controlFile[1:,5],\
                        'denL1(t-1)': controlFile[1:,6],'denL2(t-1)': controlFile[1:,7], 'denL3(t-1)': controlFile[1:,8],\
                        'denL4(t-1)': controlFile[1:,9], 'VSL1(t-1)': controlFile[1:,10], 'VSL2(t-1)': controlFile[1:,11],\
                        'TTS1(t-1)': controlFile[1:,12], 'TTS2(t-1)': controlFile[1:,13], 'TTS3(t-1)': controlFile[1:,14],\
                        'TTS4(t-1)': controlFile[1:,15], 'TTSW(t-1)': controlFile[1:,16], 'NumVeh_W': controlFile[1:,17],\
                        'A1r1': controlFile[1:,18], 'A1r2': controlFile[1:,19], 'A2r1': controlFile[1:,20],\
                        'A2r2': controlFile[1:,21], 'A1X(t)': controlFile[1:,22], 'A2X(t)': controlFile[1:,23],\
                        'V1(t)': controlFile[1:,24],'V2(t)': controlFile[1:,25],'V3(t)': controlFile[1:,26],\
                        'V4(t)': controlFile[1:,27], 'denL1(t)': controlFile[1:,28],\
                         'denL2(t)': controlFile[1:,29], 'denL3(t)': controlFile[1:,30],\
                         'denL4(t)': controlFile[1:,31], 'VSL1(t)': controlFile[1:,32],\
                        'A1LP1want': controlFile[1:,33], 'A1LP2want': controlFile[1:,34],'A1RP1want': controlFile[1:,35],\
                        'A1RP2want': controlFile[1:,36], 'A1policyWin': controlFile[1:,37],'A1LP1cont': controlFile[1:,38],\
                        'A1LP2cont': controlFile[1:,39], 'A1RP1cont': controlFile[1:,40],'A1RP2cont': controlFile[1:,41],\
                        'VSL2(t)': controlFile[1:,42], 'A2LP1want': controlFile[1:,43],'A2LP2want': controlFile[1:,44],\
                         'A2RP1want': controlFile[1:,45], 'A2RP2want': controlFile[1:,46],'A2policyWin': controlFile[1:,47],\
                         'A2LP1cont': controlFile[1:,48], 'A2LP2cont': controlFile[1:,49], 'A2RP1cont': controlFile[1:,50],\
                         'A2RP2cont': controlFile[1:,51], 'A1W_LP1': controlFile[1:,52],'A1W_LP2': controlFile[1:,53],\
                        'A1W_RP1': controlFile[1:,54],'A1W_RP2': controlFile[1:,55],'A2W_LP1': controlFile[1:,56],\
                         'A2W_LP2': controlFile[1:,57], 'A2W_RP1': controlFile[1:,58],'A2W_RP2': controlFile[1:,59],\
                        'A1alph_LP1_ai': controlFile[1:,60],'A1alph_LP2_ai': controlFile[1:,61],\
                        'A1alph_RP1_ai': controlFile[1:,62],'A1alph_RP2_ai': controlFile[1:,63],\
                         'A2alph_LP1_ai': controlFile[1:,64],'A2alph_LP2_ai': controlFile[1:,65],\
                         'A2alph_RP1_ai': controlFile[1:,66],'A2alph_RP2_ai': controlFile[1:,67],\
                        'A1alph_LP1_ak': controlFile[1:,68],'A1alph_LP2_ak': controlFile[1:,69],\
                        'A1alph_RP1_ak': controlFile[1:,70],'A1alph_RP2_ak': controlFile[1:,71],\
                         'A2alph_LP1_ak': controlFile[1:,72],'A2alph_LP2_ak': controlFile[1:,73],\
                         'A2alph_RP1_ak': controlFile[1:,74],'A2alph_RP2_ak': controlFile[1:,75],\
                        'A1alph_WLP1': controlFile[1:,76],'A1alph_WLP2': controlFile[1:,77],\
                        'A1alph_WRP1': controlFile[1:,78],'A1alph_WRP2': controlFile[1:,79],\
                         'A2alph_WLP1': controlFile[1:,80],'A2alph_WLP2': controlFile[1:,81],\
                         'A2alph_WRP1': controlFile[1:,82],'A2alph_WRP2': controlFile[1:,83],\
                        'A1alt_WLP1': controlFile[1:,84],'A1alt_WLP2': controlFile[1:,85],\
                        'A1alt_WRP1': controlFile[1:,86],'A1alt_WRP2': controlFile[1:,87],\
                         'A2alt_WLP1': controlFile[1:,88],'A2alt_WLP2': controlFile[1:,89],\
                         'A2alt_WRP1': controlFile[1:,90],'A2alt_WRP2': controlFile[1:,91],\
                        'TTSsim': controlFile[1:,92],'Occ1': controlFile[1:,93], 'Occ2': controlFile[1:,94],\
                        'Occ3': controlFile[1:,95], 'A1win': controlFile[1:,96], 'A1LPwin': controlFile[1:,97],\
                        'A1RPwin': controlFile[1:,98], 'A2win': controlFile[1:,99], 'A2LPwin': controlFile[1:,100],\
                        'A2RPwin': controlFile[1:,101], 'VSL1pos': controlFile[1:,102], 'VSL2pos': controlFile[1:,103]})
        
        df.to_excel(path_results+"NO_VSL_controlFile"+str(run)+".xlsx", index = False)


        
def rewardTTS():
    # 14.2.2020 added scale factors (alpha and beta) in the main code for two controlled cells, 
    #because they have double the length compared with cell of "congested area" (see paper)
    
    rewardNumVeh_L1 = 0
    
    rewardNumVeh_L2 = np.sum([traci.edge.getLastStepVehicleNumber("v61"),\
                            traci.edge.getLastStepVehicleNumber("v62"),\
                            traci.edge.getLastStepVehicleNumber("v63"),\
                            traci.edge.getLastStepVehicleNumber("v64"),\
                            traci.edge.getLastStepVehicleNumber("v65"),\
                            traci.edge.getLastStepVehicleNumber("v66"),\
                            traci.edge.getLastStepVehicleNumber("v67"),\
                            traci.edge.getLastStepVehicleNumber("v68"),\
                            traci.edge.getLastStepVehicleNumber("v69"),\
                            traci.lane.getLastStepVehicleNumber("v70_1"),\
                            traci.lane.getLastStepVehicleNumber("v70_2"),\
                            traci.lane.getLastStepVehicleNumber("v71_1"),\
                            traci.lane.getLastStepVehicleNumber("v71_2"),\
                            traci.lane.getLastStepVehicleNumber("v72_1"),\
                            traci.lane.getLastStepVehicleNumber("v72_2"),\
                            traci.lane.getLastStepVehicleNumber("v73_1"),\
                            traci.lane.getLastStepVehicleNumber("v73_2"),\
                            traci.lane.getLastStepVehicleNumber("v74_1"),\
                            traci.lane.getLastStepVehicleNumber("v74_2"),\
                            traci.edge.getLastStepVehicleNumber("v75"),\
                            traci.edge.getLastStepVehicleNumber("v76"),\
                            traci.edge.getLastStepVehicleNumber("v77"),\
                            traci.edge.getLastStepVehicleNumber("v78"),\
                            traci.edge.getLastStepVehicleNumber("v79"),\
                            traci.edge.getLastStepVehicleNumber("v80")])
    
    
    
    
    rewardNumVeh_L3 = np.sum([traci.edge.getLastStepVehicleNumber("v81"),\
                                traci.edge.getLastStepVehicleNumber("v82"),\
                                traci.edge.getLastStepVehicleNumber("v83"),\
                                traci.edge.getLastStepVehicleNumber("v84"),\
                                traci.edge.getLastStepVehicleNumber("v85"),\
                                traci.edge.getLastStepVehicleNumber("v86"),\
                                traci.edge.getLastStepVehicleNumber("v87"),\
                                traci.edge.getLastStepVehicleNumber("v88"),\
                                traci.edge.getLastStepVehicleNumber("v89"),\
                                traci.edge.getLastStepVehicleNumber("v90"),\
                                traci.edge.getLastStepVehicleNumber("v91"),\
                                traci.edge.getLastStepVehicleNumber("v92"),\
                                traci.edge.getLastStepVehicleNumber("v93"),\
                                traci.edge.getLastStepVehicleNumber("v94"),\
                                traci.edge.getLastStepVehicleNumber("v95"),\
                                traci.edge.getLastStepVehicleNumber("v96"),\
                                traci.edge.getLastStepVehicleNumber("v97"),\
                                traci.edge.getLastStepVehicleNumber("v98"),\
                                traci.edge.getLastStepVehicleNumber("v99"),\
                                traci.edge.getLastStepVehicleNumber("v100")])
    
    rewardNumVeh_L4 = np.sum([traci.edge.getLastStepVehicleNumber("v101"),\
                                traci.edge.getLastStepVehicleNumber("v102"),\
                                traci.edge.getLastStepVehicleNumber("v103"),\
                                traci.edge.getLastStepVehicleNumber("v104"),\
                                traci.edge.getLastStepVehicleNumber("v105"),\
                                traci.edge.getLastStepVehicleNumber("v106"),\
                                traci.lane.getLastStepVehicleNumber("v107_2"),\
                                traci.lane.getLastStepVehicleNumber("v107_3"),\
                                traci.lane.getLastStepVehicleNumber("v108_1"),\
                                traci.lane.getLastStepVehicleNumber("v108_2"),\
                                traci.lane.getLastStepVehicleNumber("v109_1"),\
                                traci.lane.getLastStepVehicleNumber("v109_2"),\
                                traci.lane.getLastStepVehicleNumber("v110_1"),\
                                traci.lane.getLastStepVehicleNumber("v110_2"),\
                                traci.lane.getLastStepVehicleNumber("v111_1"),\
                                traci.lane.getLastStepVehicleNumber("v111_2")])
   
    
    
    rewardNumVeh_W3 = np.sum([traci.lane.getLastStepHaltingNumber("18to11_0"),\
                                traci.lane.getLastStepHaltingNumber("v107_0"),\
                                traci.lane.getLastStepHaltingNumber("v108_0"),\
                                traci.lane.getLastStepHaltingNumber("v109_0"),\
                                traci.lane.getLastStepHaltingNumber("v110_0"),\
                                traci.lane.getLastStepHaltingNumber("v111_0")])
    
    return rewardNumVeh_L1, rewardNumVeh_L2, rewardNumVeh_L3, rewardNumVeh_L4, rewardNumVeh_W3

def TTS_Total():
    numberOfVeh = traci.vehicle.getIDCount()    
    return numberOfVeh


# different zones (lengths and combination) for dynamic VSL setup
def setSpeedLimit(a1, a2, VSL1pos, VSL2pos, VSL_beforeA1, VSL_beforeA2, plotHeatMap, speedLimitList):

# ========Initial speed limit for A1 section
    initSpeed = 33.33
    traci.edge.setMaxSpeed("v61", initSpeed)
    traci.edge.setMaxSpeed("v62", initSpeed)
    traci.edge.setMaxSpeed("v63", initSpeed)
    traci.edge.setMaxSpeed("v64", initSpeed)
    traci.edge.setMaxSpeed("v65", initSpeed)
    traci.edge.setMaxSpeed("v66", initSpeed)
    traci.edge.setMaxSpeed("v67", initSpeed)
    traci.edge.setMaxSpeed("v68", initSpeed)
    traci.edge.setMaxSpeed("v69", initSpeed)
    traci.lane.setMaxSpeed("v70_1", initSpeed)
    traci.lane.setMaxSpeed("v70_2", initSpeed)
    traci.lane.setMaxSpeed("v71_1", initSpeed)
    traci.lane.setMaxSpeed("v71_2", initSpeed)
    traci.lane.setMaxSpeed("v72_1", initSpeed)
    traci.lane.setMaxSpeed("v72_2", initSpeed)
    traci.lane.setMaxSpeed("v73_1", initSpeed)
    traci.lane.setMaxSpeed("v73_2", initSpeed)
    traci.lane.setMaxSpeed("v74_1", initSpeed)
    traci.lane.setMaxSpeed("v74_2", initSpeed)
    traci.lane.setMaxSpeed(":n8_1_0", initSpeed)
    traci.lane.setMaxSpeed(":n8_1_1", initSpeed)
    traci.edge.setMaxSpeed("v75", initSpeed)
    traci.edge.setMaxSpeed("v76", initSpeed)
    traci.edge.setMaxSpeed("v77", initSpeed)
    traci.edge.setMaxSpeed("v78", initSpeed)
    traci.edge.setMaxSpeed("v79", initSpeed)
    traci.edge.setMaxSpeed("v80", initSpeed)
    
    
    if plotHeatMap==1:   
        speedLimitList[61]=initSpeed
        speedLimitList[62]=initSpeed
        speedLimitList[63]=initSpeed
        speedLimitList[64]=initSpeed
        speedLimitList[65]=initSpeed
        speedLimitList[66]=initSpeed
        speedLimitList[67]=initSpeed
        speedLimitList[68]=initSpeed
        speedLimitList[69]=initSpeed
        speedLimitList[70]=initSpeed
        speedLimitList[71]=initSpeed
        speedLimitList[72]=initSpeed
        speedLimitList[73]=initSpeed
        speedLimitList[74]=initSpeed
        speedLimitList[75]=initSpeed
        speedLimitList[76]=initSpeed
        speedLimitList[77]=initSpeed
        speedLimitList[78]=initSpeed
        speedLimitList[79]=initSpeed
        speedLimitList[80]=initSpeed




# Set speed limit for specific zone A1 
    if VSL1pos==1:
        traci.edge.setMaxSpeed("v61", a1)
        traci.edge.setMaxSpeed("v62", a1)
        traci.edge.setMaxSpeed("v63", a1)
        traci.edge.setMaxSpeed("v64", a1)
        traci.edge.setMaxSpeed("v65", a1)
        traci.edge.setMaxSpeed("v66", a1)
        traci.edge.setMaxSpeed("v67", a1)
        traci.edge.setMaxSpeed("v68", a1)
        traci.edge.setMaxSpeed("v69", a1)
        traci.lane.setMaxSpeed("v70_1", a1)
        traci.lane.setMaxSpeed("v70_2", a1)
        traci.lane.setMaxSpeed("v71_1", a1)
        traci.lane.setMaxSpeed("v71_2", a1)
        traci.lane.setMaxSpeed("v72_1", a1)
        traci.lane.setMaxSpeed("v72_2", a1)
        traci.lane.setMaxSpeed("v73_1", a1)
        traci.lane.setMaxSpeed("v73_2", a1)
        traci.lane.setMaxSpeed("v74_1", a1)
        traci.lane.setMaxSpeed("v74_2", a1)
        traci.lane.setMaxSpeed(":n8_1_0", a1)
        traci.lane.setMaxSpeed(":n8_1_1", a1)
        traci.edge.setMaxSpeed("v75", a1)
        traci.edge.setMaxSpeed("v76", a1)
        traci.edge.setMaxSpeed("v77", a1)
        traci.edge.setMaxSpeed("v78", a1)
        traci.edge.setMaxSpeed("v79", a1)
        traci.edge.setMaxSpeed("v80", a1)

        if plotHeatMap==1:
            speedLimitList[61]=a1
            speedLimitList[62]=a1
            speedLimitList[63]=a1
            speedLimitList[64]=a1
            speedLimitList[65]=a1
            speedLimitList[66]=a1
            speedLimitList[67]=a1
            speedLimitList[68]=a1
            speedLimitList[69]=a1
            speedLimitList[70]=a1
            speedLimitList[71]=a1
            speedLimitList[72]=a1
            speedLimitList[73]=a1
            speedLimitList[74]=a1
            speedLimitList[75]=a1
            speedLimitList[76]=a1
            speedLimitList[77]=a1
            speedLimitList[78]=a1
            speedLimitList[79]=a1
            speedLimitList[80]=a1
        
        vehIDs_A1 = list(traci.edge.getLastStepVehicleIDs("v61")+\
                        traci.edge.getLastStepVehicleIDs("v62")+\
                        traci.edge.getLastStepVehicleIDs("v63")+\
                        traci.edge.getLastStepVehicleIDs("v64")+\
                        traci.edge.getLastStepVehicleIDs("v65")+\
                        traci.edge.getLastStepVehicleIDs("v66")+\
                        traci.edge.getLastStepVehicleIDs("v67")+\
                        traci.edge.getLastStepVehicleIDs("v68")+\
                        traci.edge.getLastStepVehicleIDs("v69")+\
                        traci.lane.getLastStepVehicleIDs("v70_1")+\
                        traci.lane.getLastStepVehicleIDs("v70_2")+\
                        traci.lane.getLastStepVehicleIDs("v71_1")+\
                        traci.lane.getLastStepVehicleIDs("v71_2")+\
                        traci.lane.getLastStepVehicleIDs("v72_1")+\
                        traci.lane.getLastStepVehicleIDs("v72_2")+\
                        traci.lane.getLastStepVehicleIDs("v73_1")+\
                        traci.lane.getLastStepVehicleIDs("v73_2")+\
                        traci.lane.getLastStepVehicleIDs("v74_1")+\
                        traci.lane.getLastStepVehicleIDs("v74_2")+\
                        traci.lane.getLastStepVehicleIDs(":n8_1_0")+\
                        traci.lane.getLastStepVehicleIDs(":n8_1_1")+\
                        traci.edge.getLastStepVehicleIDs("v75")+\
                        traci.edge.getLastStepVehicleIDs("v76")+\
                        traci.edge.getLastStepVehicleIDs("v77")+\
                        traci.edge.getLastStepVehicleIDs("v78")+\
                        traci.edge.getLastStepVehicleIDs("v79")+\
                        traci.edge.getLastStepVehicleIDs("v80"))
        

        
        
        
    elif VSL1pos==2:
        traci.edge.setMaxSpeed("v66", a1)
        traci.edge.setMaxSpeed("v67", a1)
        traci.edge.setMaxSpeed("v68", a1)
        traci.edge.setMaxSpeed("v69", a1)
        traci.lane.setMaxSpeed("v70_1", a1)
        traci.lane.setMaxSpeed("v70_2", a1)
        traci.lane.setMaxSpeed("v71_1", a1)
        traci.lane.setMaxSpeed("v71_2", a1)
        traci.lane.setMaxSpeed("v72_1", a1)
        traci.lane.setMaxSpeed("v72_2", a1)
        traci.lane.setMaxSpeed("v73_1", a1)
        traci.lane.setMaxSpeed("v73_2", a1)
        traci.lane.setMaxSpeed("v74_1", a1)
        traci.lane.setMaxSpeed("v74_2", a1)
        traci.lane.setMaxSpeed(":n8_1_0", a1)
        traci.lane.setMaxSpeed(":n8_1_1", a1)
        traci.edge.setMaxSpeed("v75", a1)
        traci.edge.setMaxSpeed("v76", a1)
        traci.edge.setMaxSpeed("v77", a1)
        traci.edge.setMaxSpeed("v78", a1)
        traci.edge.setMaxSpeed("v79", a1)
        traci.edge.setMaxSpeed("v80", a1)
        
        if plotHeatMap==1: 
            speedLimitList[66]=a1
            speedLimitList[67]=a1
            speedLimitList[68]=a1
            speedLimitList[69]=a1
            speedLimitList[70]=a1
            speedLimitList[71]=a1
            speedLimitList[72]=a1
            speedLimitList[73]=a1
            speedLimitList[74]=a1
            speedLimitList[75]=a1
            speedLimitList[76]=a1
            speedLimitList[77]=a1
            speedLimitList[78]=a1
            speedLimitList[79]=a1
            speedLimitList[80]=a1


        
        vehIDs_A1 = list(traci.edge.getLastStepVehicleIDs("v66")+\
                        traci.edge.getLastStepVehicleIDs("v67")+\
                        traci.edge.getLastStepVehicleIDs("v68")+\
                        traci.edge.getLastStepVehicleIDs("v69")+\
                        traci.lane.getLastStepVehicleIDs("v70_1")+\
                        traci.lane.getLastStepVehicleIDs("v70_2")+\
                        traci.lane.getLastStepVehicleIDs("v71_1")+\
                        traci.lane.getLastStepVehicleIDs("v71_2")+\
                        traci.lane.getLastStepVehicleIDs("v72_1")+\
                        traci.lane.getLastStepVehicleIDs("v72_2")+\
                        traci.lane.getLastStepVehicleIDs("v73_1")+\
                        traci.lane.getLastStepVehicleIDs("v73_2")+\
                        traci.lane.getLastStepVehicleIDs("v74_1")+\
                        traci.lane.getLastStepVehicleIDs("v74_2")+\
                        traci.lane.getLastStepVehicleIDs(":n8_1_0")+\
                        traci.lane.getLastStepVehicleIDs(":n8_1_1")+\
                        traci.edge.getLastStepVehicleIDs("v75")+\
                        traci.edge.getLastStepVehicleIDs("v76")+\
                        traci.edge.getLastStepVehicleIDs("v77")+\
                        traci.edge.getLastStepVehicleIDs("v78")+\
                        traci.edge.getLastStepVehicleIDs("v79")+\
                        traci.edge.getLastStepVehicleIDs("v80"))
        
    elif VSL1pos==3:
        traci.lane.setMaxSpeed("v71_1", a1)
        traci.lane.setMaxSpeed("v71_2", a1)
        traci.lane.setMaxSpeed("v72_1", a1)
        traci.lane.setMaxSpeed("v72_2", a1)
        traci.lane.setMaxSpeed("v73_1", a1)
        traci.lane.setMaxSpeed("v73_2", a1)
        traci.lane.setMaxSpeed("v74_1", a1)
        traci.lane.setMaxSpeed("v74_2", a1)
        traci.lane.setMaxSpeed(":n8_1_0", a1)
        traci.lane.setMaxSpeed(":n8_1_1", a1)
        traci.edge.setMaxSpeed("v75", a1)
        traci.edge.setMaxSpeed("v76", a1)
        traci.edge.setMaxSpeed("v77", a1)
        traci.edge.setMaxSpeed("v78", a1)
        traci.edge.setMaxSpeed("v79", a1)
        traci.edge.setMaxSpeed("v80", a1)

        if plotHeatMap==1:
            speedLimitList[71]=a1
            speedLimitList[72]=a1
            speedLimitList[73]=a1
            speedLimitList[74]=a1
            speedLimitList[75]=a1
            speedLimitList[76]=a1
            speedLimitList[77]=a1
            speedLimitList[78]=a1
            speedLimitList[79]=a1
            speedLimitList[80]=a1
        
        vehIDs_A1 = list(traci.lane.getLastStepVehicleIDs("v71_1")+\
                        traci.lane.getLastStepVehicleIDs("v71_2")+\
                        traci.lane.getLastStepVehicleIDs("v72_1")+\
                        traci.lane.getLastStepVehicleIDs("v72_2")+\
                        traci.lane.getLastStepVehicleIDs("v73_1")+\
                        traci.lane.getLastStepVehicleIDs("v73_2")+\
                        traci.lane.getLastStepVehicleIDs("v74_1")+\
                        traci.lane.getLastStepVehicleIDs("v74_2")+\
                        traci.lane.getLastStepVehicleIDs(":n8_1_0")+\
                        traci.lane.getLastStepVehicleIDs(":n8_1_1")+\
                        traci.edge.getLastStepVehicleIDs("v75")+\
                        traci.edge.getLastStepVehicleIDs("v76")+\
                        traci.edge.getLastStepVehicleIDs("v77")+\
                        traci.edge.getLastStepVehicleIDs("v78")+\
                        traci.edge.getLastStepVehicleIDs("v79")+\
                        traci.edge.getLastStepVehicleIDs("v80"))
    else:
        traci.edge.setMaxSpeed("v76", a1)
        traci.edge.setMaxSpeed("v77", a1)
        traci.edge.setMaxSpeed("v78", a1)
        traci.edge.setMaxSpeed("v79", a1)
        traci.edge.setMaxSpeed("v80", a1)
        
        if plotHeatMap==1:
            speedLimitList[76]=a1
            speedLimitList[77]=a1
            speedLimitList[78]=a1
            speedLimitList[79]=a1
            speedLimitList[80]=a1
        
        vehIDs_A1 = list(traci.edge.getLastStepVehicleIDs("v76")+\
                        traci.edge.getLastStepVehicleIDs("v77")+\
                        traci.edge.getLastStepVehicleIDs("v78")+\
                        traci.edge.getLastStepVehicleIDs("v79")+\
                        traci.edge.getLastStepVehicleIDs("v80"))
    
    

    
# ============Initial speed limit for A2 section
    traci.edge.setMaxSpeed("v81", initSpeed)
    traci.edge.setMaxSpeed("v82", initSpeed)
    traci.edge.setMaxSpeed("v83", initSpeed)
    traci.edge.setMaxSpeed("v84", initSpeed)
    traci.edge.setMaxSpeed("v85", initSpeed)
    traci.edge.setMaxSpeed("v86", initSpeed)
    traci.edge.setMaxSpeed("v87", initSpeed)
    traci.edge.setMaxSpeed("v88", initSpeed)
    traci.edge.setMaxSpeed("v89", initSpeed)
    traci.edge.setMaxSpeed("v90", initSpeed)
    traci.edge.setMaxSpeed("v91", initSpeed)
    traci.edge.setMaxSpeed("v92", initSpeed)
    traci.edge.setMaxSpeed("v93", initSpeed)
    traci.edge.setMaxSpeed("v94", initSpeed)
    traci.edge.setMaxSpeed("v95", initSpeed)
    traci.edge.setMaxSpeed("v96", initSpeed)
    traci.edge.setMaxSpeed("v97", initSpeed)
    traci.edge.setMaxSpeed("v98", initSpeed)
    traci.edge.setMaxSpeed("v99", initSpeed)
    traci.edge.setMaxSpeed("v100", initSpeed)
    
    if plotHeatMap==1:     
        speedLimitList[81]=initSpeed
        speedLimitList[82]=initSpeed
        speedLimitList[83]=initSpeed
        speedLimitList[84]=initSpeed
        speedLimitList[85]=initSpeed
        speedLimitList[86]=initSpeed
        speedLimitList[87]=initSpeed
        speedLimitList[88]=initSpeed
        speedLimitList[89]=initSpeed
        speedLimitList[90]=initSpeed
        speedLimitList[91]=initSpeed
        speedLimitList[92]=initSpeed
        speedLimitList[93]=initSpeed
        speedLimitList[94]=initSpeed
        speedLimitList[95]=initSpeed
        speedLimitList[96]=initSpeed
        speedLimitList[97]=initSpeed
        speedLimitList[98]=initSpeed
        speedLimitList[99]=initSpeed
        speedLimitList[100]=initSpeed
    
# Set speed limit for specific zone A2
    if VSL2pos==1:
        traci.edge.setMaxSpeed("v81", a2)
        traci.edge.setMaxSpeed("v82", a2)
        traci.edge.setMaxSpeed("v83", a2)
        traci.edge.setMaxSpeed("v84", a2)
        traci.edge.setMaxSpeed("v85", a2)
        traci.edge.setMaxSpeed("v86", a2)
        traci.edge.setMaxSpeed("v87", a2)
        traci.edge.setMaxSpeed("v88", a2)
        traci.edge.setMaxSpeed("v89", a2)
        traci.edge.setMaxSpeed("v90", a2)
        traci.edge.setMaxSpeed("v91", a2)
        traci.edge.setMaxSpeed("v92", a2)
        traci.edge.setMaxSpeed("v93", a2)
        traci.edge.setMaxSpeed("v94", a2)
        traci.edge.setMaxSpeed("v95", a2)
        traci.edge.setMaxSpeed("v96", a2)
        traci.edge.setMaxSpeed("v97", a2)
        traci.edge.setMaxSpeed("v98", a2)
        traci.edge.setMaxSpeed("v99", a2)
        traci.edge.setMaxSpeed("v100", a2)
        
        if plotHeatMap==1:     
            speedLimitList[81]=a2
            speedLimitList[82]=a2
            speedLimitList[83]=a2
            speedLimitList[84]=a2
            speedLimitList[85]=a2
            speedLimitList[86]=a2
            speedLimitList[87]=a2
            speedLimitList[88]=a2
            speedLimitList[89]=a2
            speedLimitList[90]=a2
            speedLimitList[91]=a2
            speedLimitList[92]=a2
            speedLimitList[93]=a2
            speedLimitList[94]=a2
            speedLimitList[95]=a2
            speedLimitList[96]=a2
            speedLimitList[97]=a2
            speedLimitList[98]=a2
            speedLimitList[99]=a2
            speedLimitList[100]=a2
        
        vehIDs_A2 = list(traci.edge.getLastStepVehicleIDs("v81")+\
                        traci.edge.getLastStepVehicleIDs("v82")+\
                        traci.edge.getLastStepVehicleIDs("v83")+\
                        traci.edge.getLastStepVehicleIDs("v84")+\
                        traci.edge.getLastStepVehicleIDs("v85")+\
                        traci.edge.getLastStepVehicleIDs("v86")+\
                        traci.edge.getLastStepVehicleIDs("v87")+\
                        traci.edge.getLastStepVehicleIDs("v88")+\
                        traci.edge.getLastStepVehicleIDs("v89")+\
                        traci.edge.getLastStepVehicleIDs("v90")+\
                        traci.edge.getLastStepVehicleIDs("v91")+\
                        traci.edge.getLastStepVehicleIDs("v92")+\
                        traci.edge.getLastStepVehicleIDs("v93")+\
                        traci.edge.getLastStepVehicleIDs("v94")+\
                        traci.edge.getLastStepVehicleIDs("v95")+\
                        traci.edge.getLastStepVehicleIDs("v96")+\
                        traci.edge.getLastStepVehicleIDs("v97")+\
                        traci.edge.getLastStepVehicleIDs("v98")+\
                        traci.edge.getLastStepVehicleIDs("v99")+\
                        traci.edge.getLastStepVehicleIDs("v100"))
        
        

        
        
        
    elif VSL2pos==2:
        traci.edge.setMaxSpeed("v81", a2)
        traci.edge.setMaxSpeed("v82", a2)
        traci.edge.setMaxSpeed("v83", a2)
        traci.edge.setMaxSpeed("v84", a2)
        traci.edge.setMaxSpeed("v85", a2)
        traci.edge.setMaxSpeed("v86", a2)
        traci.edge.setMaxSpeed("v87", a2)
        traci.edge.setMaxSpeed("v88", a2)
        traci.edge.setMaxSpeed("v89", a2)
        traci.edge.setMaxSpeed("v90", a2)
        traci.edge.setMaxSpeed("v91", a2)
        traci.edge.setMaxSpeed("v92", a2)
        traci.edge.setMaxSpeed("v93", a2)
        traci.edge.setMaxSpeed("v94", a2)
        traci.edge.setMaxSpeed("v95", a2)

        if plotHeatMap==1:     
            speedLimitList[81]=a2
            speedLimitList[82]=a2
            speedLimitList[83]=a2
            speedLimitList[84]=a2
            speedLimitList[85]=a2
            speedLimitList[86]=a2
            speedLimitList[87]=a2
            speedLimitList[88]=a2
            speedLimitList[89]=a2
            speedLimitList[90]=a2
            speedLimitList[91]=a2
            speedLimitList[92]=a2
            speedLimitList[93]=a2
            speedLimitList[94]=a2
            speedLimitList[95]=a2

        
        vehIDs_A2 = list(traci.edge.getLastStepVehicleIDs("v81")+\
                        traci.edge.getLastStepVehicleIDs("v82")+\
                        traci.edge.getLastStepVehicleIDs("v83")+\
                        traci.edge.getLastStepVehicleIDs("v84")+\
                        traci.edge.getLastStepVehicleIDs("v85")+\
                        traci.edge.getLastStepVehicleIDs("v86")+\
                        traci.edge.getLastStepVehicleIDs("v87")+\
                        traci.edge.getLastStepVehicleIDs("v88")+\
                        traci.edge.getLastStepVehicleIDs("v89")+\
                        traci.edge.getLastStepVehicleIDs("v90")+\
                        traci.edge.getLastStepVehicleIDs("v91")+\
                        traci.edge.getLastStepVehicleIDs("v92")+\
                        traci.edge.getLastStepVehicleIDs("v93")+\
                        traci.edge.getLastStepVehicleIDs("v94")+\
                        traci.edge.getLastStepVehicleIDs("v95"))
        
    elif VSL2pos==3:
        traci.edge.setMaxSpeed("v81", a2)
        traci.edge.setMaxSpeed("v82", a2)
        traci.edge.setMaxSpeed("v83", a2)
        traci.edge.setMaxSpeed("v84", a2)
        traci.edge.setMaxSpeed("v85", a2)
        traci.edge.setMaxSpeed("v86", a2)
        traci.edge.setMaxSpeed("v87", a2)
        traci.edge.setMaxSpeed("v88", a2)
        traci.edge.setMaxSpeed("v89", a2)
        traci.edge.setMaxSpeed("v90", a2)
        
        if plotHeatMap==1:     
            speedLimitList[81]=a2
            speedLimitList[82]=a2
            speedLimitList[83]=a2
            speedLimitList[84]=a2
            speedLimitList[85]=a2
            speedLimitList[86]=a2
            speedLimitList[87]=a2
            speedLimitList[88]=a2
            speedLimitList[89]=a2
            speedLimitList[90]=a2

            
        vehIDs_A2 = list(traci.edge.getLastStepVehicleIDs("v81")+\
                        traci.edge.getLastStepVehicleIDs("v82")+\
                        traci.edge.getLastStepVehicleIDs("v83")+\
                        traci.edge.getLastStepVehicleIDs("v84")+\
                        traci.edge.getLastStepVehicleIDs("v85")+\
                        traci.edge.getLastStepVehicleIDs("v86")+\
                        traci.edge.getLastStepVehicleIDs("v87")+\
                        traci.edge.getLastStepVehicleIDs("v88")+\
                        traci.edge.getLastStepVehicleIDs("v89")+\
                        traci.edge.getLastStepVehicleIDs("v90"))
    else:
        traci.edge.setMaxSpeed("v81", a2)
        traci.edge.setMaxSpeed("v82", a2)
        traci.edge.setMaxSpeed("v83", a2)
        traci.edge.setMaxSpeed("v84", a2)
        traci.edge.setMaxSpeed("v85", a2)

        if plotHeatMap==1:     
            speedLimitList[81]=a2
            speedLimitList[82]=a2
            speedLimitList[83]=a2
            speedLimitList[84]=a2
            speedLimitList[85]=a2
        
        
        vehIDs_A2 = list(traci.edge.getLastStepVehicleIDs("v81")+\
                        traci.edge.getLastStepVehicleIDs("v82")+\
                        traci.edge.getLastStepVehicleIDs("v83")+\
                        traci.edge.getLastStepVehicleIDs("v84")+\
                        traci.edge.getLastStepVehicleIDs("v85"))
        
# the more realistic VSL application at specific points instead of along an edge
# allow us to simulate point VSL system (like VMS at specific location on motorways)
# if we just change the speed on edge at time (t) it influences all vehicles on that edge which is not realistic for current traffic
# since vehicles that have passed the VMS sign will continue traveling with the mandatory speed limit from the control time step (t-1)
# THIS IS ACHIEVED BY ADJUSTING speedFactor for vehicles positioned on edge with VSL each time speed limit gets changed

    SF = 1 # perfect speedFactor
    sF_org_L1_list = []
    VSL_diff_L1 = ((VSL_beforeA1[0]-VSL_beforeA1[1])/VSL_beforeA1[1])
    for idL1 in vehIDs_A1:
        sF_diff_L1 = (VSL_diff_L1 - VSL_diff_L1*(SF - traci.vehicle.getSpeedFactor(idL1)))
        sF_org_L1_list.append(traci.vehicle.getSpeedFactor(idL1))
        d1=traci.vehicle.getSpeedFactor(idL1) + sF_diff_L1
        traci.vehicle.setSpeedFactor(idL1, d1)

                                     
    sF_org_L2_list = []
    VSL_diff_L2 = ((VSL_beforeA2[0]-VSL_beforeA2[1])/VSL_beforeA2[1])            
    for idL2 in vehIDs_A2:
        sF_diff_L2 = (VSL_diff_L2 - VSL_diff_L2*(SF - traci.vehicle.getSpeedFactor(idL2)))
        sF_org_L2_list.append(traci.vehicle.getSpeedFactor(idL2))
        d2=traci.vehicle.getSpeedFactor(idL2) + sF_diff_L2
        traci.vehicle.setSpeedFactor(idL2, d2)

                                     
    return vehIDs_A1, vehIDs_A2, sF_org_L1_list, sF_org_L2_list, speedLimitList


def setSpeedFactorL1(vehIDs, agent, sF_org_list, VSL1pos):
    if VSL1pos==1:
        intersectionA1 = set(vehIDs).intersection\
                        (traci.lane.getLastStepVehicleIDs("v70_0")+\
                         traci.lane.getLastStepVehicleIDs("v71_0")+\
                         traci.lane.getLastStepVehicleIDs("v72_0")+\
                         traci.edge.getLastStepVehicleIDs("v81")+\
                         traci.edge.getLastStepVehicleIDs("v82")+\
                         traci.edge.getLastStepVehicleIDs("v83"))
    elif VSL1pos==2:
        intersectionA1 = set(vehIDs).intersection\
                        (traci.lane.getLastStepVehicleIDs("v70_0")+\
                         traci.lane.getLastStepVehicleIDs("v71_0")+\
                         traci.lane.getLastStepVehicleIDs("v72_0")+\
                         traci.edge.getLastStepVehicleIDs("v81")+\
                         traci.edge.getLastStepVehicleIDs("v82")+\
                         traci.edge.getLastStepVehicleIDs("v83"))
    elif VSL1pos==3:
        intersectionA1 = set(vehIDs).intersection\
                        (traci.lane.getLastStepVehicleIDs("v71_0")+\
                         traci.lane.getLastStepVehicleIDs("v72_0")+\
                         traci.edge.getLastStepVehicleIDs("v81")+\
                         traci.edge.getLastStepVehicleIDs("v82")+\
                         traci.edge.getLastStepVehicleIDs("v83"))
    else:
        intersectionA1 = set(vehIDs).intersection\
                        (traci.edge.getLastStepVehicleIDs("v81")+\
                         traci.edge.getLastStepVehicleIDs("v82")+\
                         traci.edge.getLastStepVehicleIDs("v83"))
        
    if len(list(intersectionA1)) != 0:
        for idL1 in list(intersectionA1):
            indexL1 = vehIDs.index(idL1)
            traci.vehicle.setSpeedFactor(vehIDs[indexL1], sF_org_list[indexL1])
        for idL1 in list(intersectionA1):
            indexL1 = vehIDs.index(idL1)
            del vehIDs[indexL1]
            del sF_org_list[indexL1]
        vehIDs_A1 = vehIDs
        sF_org_L1_list = sF_org_list
    else:
        vehIDs_A1 = vehIDs
        sF_org_L1_list = sF_org_list

def setSpeedFactorL2(vehIDs, agent, sF_org_list, VSL2pos):
    if VSL2pos==1:
        intersectionA2 = set(vehIDs).intersection\
            (traci.edge.getLastStepVehicleIDs("v101")+\
             traci.edge.getLastStepVehicleIDs("v102")+\
             traci.edge.getLastStepVehicleIDs("v103"))
        
    elif VSL2pos==2:
        intersectionA2 = set(vehIDs).intersection\
                (traci.edge.getLastStepVehicleIDs("v96")+\
                 traci.edge.getLastStepVehicleIDs("v97")+\
                 traci.edge.getLastStepVehicleIDs("v98"))
        
    elif VSL2pos==3:
        intersectionA2 = set(vehIDs).intersection\
                (traci.edge.getLastStepVehicleIDs("v91")+\
                 traci.edge.getLastStepVehicleIDs("v92")+\
                 traci.edge.getLastStepVehicleIDs("v93"))
        
    else:
        intersectionA2 = set(vehIDs).intersection\
                (traci.edge.getLastStepVehicleIDs("v86")+\
                 traci.edge.getLastStepVehicleIDs("v87")+\
                 traci.edge.getLastStepVehicleIDs("v88"))
        
    if len(list(intersectionA2)) != 0:
        for idL2 in list(intersectionA2): 
            indexL2 = vehIDs.index(idL2)
            traci.vehicle.setSpeedFactor(vehIDs[indexL2], sF_org_list[indexL2])
        for idL2 in list(intersectionA2):
            indexL2 = vehIDs.index(idL2)
            del vehIDs[indexL2]
            del sF_org_list[indexL2]
        vehIDs_A2 = vehIDs
        sF_org_L2_list = sF_org_list
    else:
        vehIDs_A2 = vehIDs
        sF_org_L2_list = sF_org_list



def current_Density_L1L2L3L4(numLanes):
    
    currentDensityL1 = 0
    
    currentDensityL2 = (1/(1000/1000))*(1/numLanes)*np.sum([traci.edge.getLastStepVehicleNumber("v61"),\
                                                            traci.edge.getLastStepVehicleNumber("v62"),\
                                                            traci.edge.getLastStepVehicleNumber("v63"),\
                                                            traci.edge.getLastStepVehicleNumber("v64"),\
                                                            traci.edge.getLastStepVehicleNumber("v65"),\
                                                            traci.edge.getLastStepVehicleNumber("v66"),\
                                                            traci.edge.getLastStepVehicleNumber("v67"),\
                                                            traci.edge.getLastStepVehicleNumber("v68"),\
                                                            traci.edge.getLastStepVehicleNumber("v69"),\
                                                            traci.lane.getLastStepVehicleNumber("v70_1"),\
                                                            traci.lane.getLastStepVehicleNumber("v70_2"),\
                                                            traci.lane.getLastStepVehicleNumber("v71_1"),\
                                                            traci.lane.getLastStepVehicleNumber("v71_2"),\
                                                            traci.lane.getLastStepVehicleNumber("v72_1"),\
                                                            traci.lane.getLastStepVehicleNumber("v72_2"),\
                                                            traci.lane.getLastStepVehicleNumber("v73_1"),\
                                                            traci.lane.getLastStepVehicleNumber("v73_2"),\
                                                            traci.lane.getLastStepVehicleNumber("v74_1"),\
                                                            traci.lane.getLastStepVehicleNumber("v74_2"),\
                                                            traci.edge.getLastStepVehicleNumber("v75"),\
                                                            traci.edge.getLastStepVehicleNumber("v76"),\
                                                            traci.edge.getLastStepVehicleNumber("v77"),\
                                                            traci.edge.getLastStepVehicleNumber("v78"),\
                                                            traci.edge.getLastStepVehicleNumber("v79"),\
                                                            traci.edge.getLastStepVehicleNumber("v80")])
    
    

    
    
    
    currentDensityL3 = (1/(1000/1000))*(1/numLanes)*np.sum([traci.edge.getLastStepVehicleNumber("v81"),\
                                                            traci.edge.getLastStepVehicleNumber("v82"),\
                                                            traci.edge.getLastStepVehicleNumber("v83"),\
                                                            traci.edge.getLastStepVehicleNumber("v84"),\
                                                            traci.edge.getLastStepVehicleNumber("v85"),\
                                                            traci.edge.getLastStepVehicleNumber("v86"),\
                                                            traci.edge.getLastStepVehicleNumber("v87"),\
                                                            traci.edge.getLastStepVehicleNumber("v88"),\
                                                            traci.edge.getLastStepVehicleNumber("v89"),\
                                                            traci.edge.getLastStepVehicleNumber("v90"),\
                                                            traci.edge.getLastStepVehicleNumber("v91"),\
                                                            traci.edge.getLastStepVehicleNumber("v92"),\
                                                            traci.edge.getLastStepVehicleNumber("v93"),\
                                                            traci.edge.getLastStepVehicleNumber("v94"),\
                                                            traci.edge.getLastStepVehicleNumber("v95"),\
                                                            traci.edge.getLastStepVehicleNumber("v96"),\
                                                            traci.edge.getLastStepVehicleNumber("v97"),\
                                                            traci.edge.getLastStepVehicleNumber("v98"),\
                                                            traci.edge.getLastStepVehicleNumber("v99"),\
                                                            traci.edge.getLastStepVehicleNumber("v100")])
    
    

    
    
    currentDensityL4 = (1/(550/1000))*(1/numLanes)*(np.sum([traci.edge.getLastStepVehicleNumber("v101"),\
                                                            traci.edge.getLastStepVehicleNumber("v102"),\
                                                            traci.edge.getLastStepVehicleNumber("v103"),\
                                                            traci.edge.getLastStepVehicleNumber("v104"),\
                                                            traci.edge.getLastStepVehicleNumber("v105"),\
                                                            traci.edge.getLastStepVehicleNumber("v106"),\
                                                            traci.lane.getLastStepVehicleNumber("v107_2"),\
                                                            traci.lane.getLastStepVehicleNumber("v107_3"),\
                                                            traci.lane.getLastStepVehicleNumber("v108_1"),\
                                                            traci.lane.getLastStepVehicleNumber("v108_2"),\
                                                            traci.lane.getLastStepVehicleNumber("v109_1"),\
                                                            traci.lane.getLastStepVehicleNumber("v109_2"),\
                                                            traci.lane.getLastStepVehicleNumber("v110_1"),\
                                                            traci.lane.getLastStepVehicleNumber("v110_2"),\
                                                            traci.lane.getLastStepVehicleNumber("v111_1"),\
                                                            traci.lane.getLastStepVehicleNumber("v111_2")]))
    
    return currentDensityL1, currentDensityL2, currentDensityL3, currentDensityL4

# we divided the entire model into smaller segments in which macroscopic parameter (speed) is used for the heat map plot 
def speedPerSegments():
    speedList=[]
    for i in range(1,161):
        speedList.append(traci.edge.getLastStepMeanSpeed("v"+str(i)))
    return speedList

def currentSpeed_L1L2L3L4():
            speedL1 = 0 
            speedL2 = np.mean([traci.edge.getLastStepMeanSpeed("v61"),\
                                traci.edge.getLastStepMeanSpeed("v62"),\
                                traci.edge.getLastStepMeanSpeed("v63"),\
                                traci.edge.getLastStepMeanSpeed("v64"),\
                                traci.edge.getLastStepMeanSpeed("v65"),\
                                traci.edge.getLastStepMeanSpeed("v66"),\
                                traci.edge.getLastStepMeanSpeed("v67"),\
                                traci.edge.getLastStepMeanSpeed("v68"),\
                                traci.edge.getLastStepMeanSpeed("v69"),\
                                traci.lane.getLastStepMeanSpeed("v70_1"),\
                                traci.lane.getLastStepMeanSpeed("v70_2"),\
                                traci.lane.getLastStepMeanSpeed("v71_1"),\
                                traci.lane.getLastStepMeanSpeed("v71_2"),\
                                traci.lane.getLastStepMeanSpeed("v72_1"),\
                                traci.lane.getLastStepMeanSpeed("v72_2"),\
                                traci.lane.getLastStepMeanSpeed("v73_1"),\
                                traci.lane.getLastStepMeanSpeed("v73_2"),\
                                traci.lane.getLastStepMeanSpeed("v74_1"),\
                                traci.lane.getLastStepMeanSpeed("v74_2"),\
                                traci.edge.getLastStepMeanSpeed("v75"),\
                                traci.edge.getLastStepMeanSpeed("v76"),\
                                traci.edge.getLastStepMeanSpeed("v77"),\
                                traci.edge.getLastStepMeanSpeed("v78"),\
                                traci.edge.getLastStepMeanSpeed("v79"),\
                                traci.edge.getLastStepMeanSpeed("v80")])
            
            speedL3 = np.mean([traci.edge.getLastStepMeanSpeed("v81"),\
                                traci.edge.getLastStepMeanSpeed("v82"),\
                                traci.edge.getLastStepMeanSpeed("v83"),\
                                traci.edge.getLastStepMeanSpeed("v84"),\
                                traci.edge.getLastStepMeanSpeed("v85"),\
                                traci.edge.getLastStepMeanSpeed("v86"),\
                                traci.edge.getLastStepMeanSpeed("v87"),\
                                traci.edge.getLastStepMeanSpeed("v88"),\
                                traci.edge.getLastStepMeanSpeed("v89"),\
                                traci.edge.getLastStepMeanSpeed("v90"),\
                                traci.edge.getLastStepMeanSpeed("v91"),\
                                traci.edge.getLastStepMeanSpeed("v92"),\
                                traci.edge.getLastStepMeanSpeed("v93"),\
                                traci.edge.getLastStepMeanSpeed("v94"),\
                                traci.edge.getLastStepMeanSpeed("v95"),\
                                traci.edge.getLastStepMeanSpeed("v96"),\
                                traci.edge.getLastStepMeanSpeed("v97"),\
                                traci.edge.getLastStepMeanSpeed("v98"),\
                                traci.edge.getLastStepMeanSpeed("v99"),\
                                traci.edge.getLastStepMeanSpeed("v100")])
            
            
            # 12.2.2020 avg. speed Cell 3 (on-ramp area)
            speedL4 = np.mean([traci.edge.getLastStepMeanSpeed("v101"),\
                                traci.edge.getLastStepMeanSpeed("v102"),\
                                traci.edge.getLastStepMeanSpeed("v103"),\
                                traci.edge.getLastStepMeanSpeed("v104"),\
                                traci.edge.getLastStepMeanSpeed("v105"),\
                                traci.edge.getLastStepMeanSpeed("v106"),\
                                traci.lane.getLastStepMeanSpeed("v107_2"),\
                                traci.lane.getLastStepMeanSpeed("v107_3"),\
                                traci.lane.getLastStepMeanSpeed("v108_1"),\
                                traci.lane.getLastStepMeanSpeed("v108_2"),\
                                traci.lane.getLastStepMeanSpeed("v109_1"),\
                                traci.lane.getLastStepMeanSpeed("v109_2"),\
                                traci.lane.getLastStepMeanSpeed("v110_1"),\
                                traci.lane.getLastStepMeanSpeed("v110_2"),\
                                traci.lane.getLastStepMeanSpeed("v111_1"),\
                                traci.lane.getLastStepMeanSpeed("v111_2")])
            
            return speedL1, speedL2, speedL3, speedL4


def initial_Lane_Atributes(accel_deccel_off_on_ramps,ramps,disallowed):
    
    traci.lane.setMaxSpeed(accel_deccel_off_on_ramps[0], 28.0) # 22 m/s = 80 km/h
    traci.lane.setMaxSpeed(accel_deccel_off_on_ramps[1], 28.0) # 36 m/s = 130 km/h
    traci.lane.setMaxSpeed(accel_deccel_off_on_ramps[2], 28.0)
    traci.lane.setMaxSpeed(accel_deccel_off_on_ramps[3], 28.0)
    traci.lane.setMaxSpeed(accel_deccel_off_on_ramps[4], 28.0)
    traci.lane.setMaxSpeed(accel_deccel_off_on_ramps[5], 28.0)
    traci.lane.setMaxSpeed(accel_deccel_off_on_ramps[6], 28.0)
    traci.lane.setMaxSpeed(accel_deccel_off_on_ramps[7], 28.0)
    traci.lane.setMaxSpeed(accel_deccel_off_on_ramps[8], 28.0)
    traci.lane.setMaxSpeed(accel_deccel_off_on_ramps[9], 28.0)
    traci.lane.setMaxSpeed(accel_deccel_off_on_ramps[10], 28.0)
    traci.lane.setMaxSpeed(accel_deccel_off_on_ramps[11], 28.0)
    traci.lane.setMaxSpeed(accel_deccel_off_on_ramps[12], 28.0)
    traci.lane.setMaxSpeed(accel_deccel_off_on_ramps[13], 28.0)
    traci.lane.setMaxSpeed(accel_deccel_off_on_ramps[14], 28.0)

    traci.lane.setMaxSpeed(ramps[0], 16.0) # 11 m/s = 40 km/h
    traci.lane.setMaxSpeed(ramps[1], 16.0) # 16 m/s = 60 km/h
    traci.lane.setMaxSpeed(ramps[2], 16.0)

    traci.lane.setDisallowed(disallowed[0], "all")
    traci.lane.setDisallowed(disallowed[1], "all")
    
def occupancyOnRamp():
    
    occupancyLinkRamp = traci.lane.getLastStepOccupancy("18to11_0")
    occupancyAccelLane1 = traci.lane.getLastStepOccupancy("v107_0") + traci.lane.getLastStepOccupancy("v108_0")
    occupancyAccelLane2 = traci.lane.getLastStepOccupancy("v109_0")+ traci.lane.getLastStepOccupancy("v110_0")+\
                            traci.lane.getLastStepOccupancy("v111_0")

    return [occupancyLinkRamp, occupancyAccelLane1, occupancyAccelLane2]
    
    
    
#===== run simulation and traine VSL agents
run=0
end_learning=14000
exploit=100 # #of sumulations for analysis
vsl_on_off=1 # (1 active VSL)
load=0   #load=run-1 # (# load old Qs and Ws, 0 start learning from scratch)
update_QW = 1 # 0 - no update, 1 - update
plotHeatMap = 0

time_sample_TTSreward = 5
time_res_sec=50
control_time_perid = 150 # sec
numberSaveData = 20 # number of simulations after which data must be saved (you can change this e.g., 40 which might speed up the process)
                      #(safety property, if crashes, learning parameters can be re-loaded from the last saved sim)

# init. some parameters in SUMO network 
accel_deccel_off_on_ramps = ["v107_0","v108_0", "v109_0", "v110_0", "v111_0",\
                             "v47_0", "v48_0", "v49_0", "v50_0", "v51_0",\
                             "v70_0", "v71_0", "v72_0", "v73_0", "v74_0"]
ramps = ["16to4_0", "8to17_0", "18to11_0"]
disallowed = ["v107_1", "v47_1"] # "All"

# run simulations
while(run<end_learning+exploit+2):
    
    if run == 0:
        t0=datetime.datetime.now()
    elif run == 1:
        print(datetime.datetime.now()-t0)
            
    if run<end_learning+1:
        epsilon=math.exp((((-1)*math.log(20))/6000)*run)
    else:
        epsilon=0
        plotHeatMap = 1
        
    if run==end_learning+exploit+1:
        vsl_on_off=0
    
    # load old knowledge in case! 
    if load!=0 and vsl_on_off==1:
        A1.Q_LPi[0] = np.loadtxt(open(path_results+'A1LP1'+str(load)+'.csv', 'rt'), delimiter=",")
        A2.Q_LPi[0] = np.loadtxt(open(path_results+'A2LP1'+str(load)+'.csv', 'rt'), delimiter=",")
        A1.Q_RPi[0] = np.loadtxt(open(path_results+'A1RP1'+str(load)+'.csv', 'rt'), delimiter=",")
        A2.Q_RPi[0] = np.loadtxt(open(path_results+'A2RP1'+str(load)+'.csv', 'rt'), delimiter=",")
        A1.Q_LPi[1] = np.loadtxt(open(path_results+'A1LP2'+str(load)+'.csv', 'rt'), delimiter=",")
        A2.Q_LPi[1] = np.loadtxt(open(path_results+'A2LP2'+str(load)+'.csv', 'rt'), delimiter=",")
        A1.Q_RPi[1] = np.loadtxt(open(path_results+'A1RP2'+str(load)+'.csv', 'rt'), delimiter=",")
        A2.Q_RPi[1] = np.loadtxt(open(path_results+'A2RP2'+str(load)+'.csv', 'rt'), delimiter=",")
        A1.W_LPi[0] = np.resize(np.loadtxt(open(path_results+'A1W_LP1'+str(load)+'.csv', 'rt'), delimiter=","), 4608).reshape(4608,1)
        A2.W_LPi[0] = np.resize(np.loadtxt(open(path_results+'A2W_LP1'+str(load)+'.csv', 'rt'), delimiter=","), 4608).reshape(4608,1)
        A1.W_RPi[0] = np.resize(np.loadtxt(open(path_results+'A1W_RP1'+str(load)+'.csv', 'rt'), delimiter=","), 4608).reshape(4608,1)
        A2.W_RPi[0] = np.resize(np.loadtxt(open(path_results+'A2W_RP1'+str(load)+'.csv', 'rt'), delimiter=","), 4608).reshape(4608,1)
        A1.W_LPi[1] = np.resize(np.loadtxt(open(path_results+'A1W_LP2'+str(load)+'.csv', 'rt'), delimiter=","), 4608).reshape(4608,1)
        A2.W_LPi[1] = np.resize(np.loadtxt(open(path_results+'A2W_LP2'+str(load)+'.csv', 'rt'), delimiter=","), 4608).reshape(4608,1)
        A1.W_RPi[1] = np.resize(np.loadtxt(open(path_results+'A1W_RP2'+str(load)+'.csv', 'rt'), delimiter=","), 4608).reshape(4608,1)
        A2.W_RPi[1] = np.resize(np.loadtxt(open(path_results+'A2W_RP2'+str(load)+'.csv', 'rt'), delimiter=","), 4608).reshape(4608,1)
             
        A1.Num_Visited_LPi_x_ak[0] = np.loadtxt(open(path_results+'A1NumLP1_X_ak'+str(load)+'.csv', 'rt'), delimiter=',')
        A2.Num_Visited_LPi_x_ak[0] = np.loadtxt(open(path_results+'A2NumLP1_X_ak'+str(load)+'.csv', 'rt'), delimiter=',')
        A1.Num_Visited_RPi_x_ak[0] = np.loadtxt(open(path_results+'A1NumRP1_X_ak'+str(load)+'.csv', 'rt'), delimiter=',')
        A2.Num_Visited_RPi_x_ak[0] = np.loadtxt(open(path_results+'A2NumRP1_X_ak'+str(load)+'.csv', 'rt'), delimiter=',')
        A1.Num_Visited_LPi_x_ak[1] = np.loadtxt(open(path_results+'A1NumLP2_X_ak'+str(load)+'.csv', 'rt'), delimiter=',')
        A2.Num_Visited_LPi_x_ak[1] = np.loadtxt(open(path_results+'A2NumLP2_X_ak'+str(load)+'.csv', 'rt'), delimiter=',')
        A1.Num_Visited_RPi_x_ak[1] = np.loadtxt(open(path_results+'A1NumRP2_X_ak'+str(load)+'.csv', 'rt'), delimiter=',')
        A2.Num_Visited_RPi_x_ak[1] = np.loadtxt(open(path_results+'A2NumRP2_X_ak'+str(load)+'.csv', 'rt'), delimiter=',')
       
        A1.Num_Visited_LPi_x[0] = np.resize(np.loadtxt(open(path_results+'A1NumLP1_X'+str(load)+'.csv', 'rt'), delimiter=','), 4608).reshape(4608,1)
        A2.Num_Visited_LPi_x[0] = np.resize(np.loadtxt(open(path_results+'A2NumLP1_X'+str(load)+'.csv', 'rt'), delimiter=','), 4608).reshape(4608,1)
        A1.Num_Visited_RPi_x[0] = np.resize(np.loadtxt(open(path_results+'A1NumRP1_X'+str(load)+'.csv', 'rt'), delimiter=','), 4608).reshape(4608,1)
        A2.Num_Visited_RPi_x[0] = np.resize(np.loadtxt(open(path_results+'A2NumRP1_X'+str(load)+'.csv', 'rt'), delimiter=','), 4608).reshape(4608,1)
        A1.Num_Visited_LPi_x[1] = np.resize(np.loadtxt(open(path_results+'A1NumLP2_X'+str(load)+'.csv', 'rt'), delimiter=','), 4608).reshape(4608,1)
        A2.Num_Visited_LPi_x[1] = np.resize(np.loadtxt(open(path_results+'A2NumLP2_X'+str(load)+'.csv', 'rt'), delimiter=','), 4608).reshape(4608,1)
        A1.Num_Visited_RPi_x[1] = np.resize(np.loadtxt(open(path_results+'A1NumRP2_X'+str(load)+'.csv', 'rt'), delimiter=','), 4608).reshape(4608,1)
        A2.Num_Visited_RPi_x[1] = np.resize(np.loadtxt(open(path_results+'A2NumRP2_X'+str(load)+'.csv', 'rt'), delimiter=','), 4608).reshape(4608,1)
       
        A1.Num_Visited_LPi_x_ai[0] = np.loadtxt(open(path_results+'A1NumLP1_X_ai'+str(load)+'.csv', 'rt'), delimiter=',')
        A2.Num_Visited_LPi_x_ai[0] = np.loadtxt(open(path_results+'A2NumLP1_X_ai'+str(load)+'.csv', 'rt'), delimiter=',')
        A1.Num_Visited_RPi_x_ai[0] = np.loadtxt(open(path_results+'A1NumRP1_X_ai'+str(load)+'.csv', 'rt'), delimiter=',')
        A2.Num_Visited_RPi_x_ai[0] = np.loadtxt(open(path_results+'A2NumRP1_X_ai'+str(load)+'.csv', 'rt'), delimiter=',')
        A1.Num_Visited_LPi_x_ai[1] = np.loadtxt(open(path_results+'A1NumLP2_X_ai'+str(load)+'.csv', 'rt'), delimiter=',')
        A2.Num_Visited_LPi_x_ai[1] = np.loadtxt(open(path_results+'A2NumLP2_X_ai'+str(load)+'.csv', 'rt'), delimiter=',')
        A1.Num_Visited_RPi_x_ai[1] = np.loadtxt(open(path_results+'A1NumRP2_X_ai'+str(load)+'.csv', 'rt'), delimiter=',')
        A2.Num_Visited_RPi_x_ai[1] = np.loadtxt(open(path_results+'A2NumRP2_X_ai'+str(load)+'.csv', 'rt'), delimiter=',')

        load=0

    traci.start(sumoCmd)
    step = 0
    

    initial_Lane_Atributes(accel_deccel_off_on_ramps,ramps,disallowed)
    speedCellL1 = np.array([0.0,0.0,0.0,0.0])
    speedCellL2 = np.array([0.0,0.0,0.0,0.0])
    speedCellL3 = np.array([0.0,0.0,0.0,0.0])
    speedCellL4 = np.array([0.0,0.0,0.0,0.0])

    currentDensityOnRamp = 0
    speedOnVMS = np.array([33.33, 33.33, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    controlFile = np.zeros((1,104))

    
    speedLimitList=[]
    for ll in range(0,160):
        speedLimitList.append(33.33)
    speedHeatMap=np.zeros((1,160))
    speedLimitHeatMap=np.zeros((1,160))

    
    densityL1Steps = np.array([0.0,0.0,0.0,0.0])
    densityL2Steps = np.array([0.0,0.0,0.0,0.0])
    densityL3Steps = np.array([0.0,0.0,0.0,0.0])
    densityL4Steps = np.array([0.0,0.0,0.0,0.0])
    


    TTS_Sim=0
    TTS = 0
    TTS1 = 0
    TTS2 = 0
    TTS3 = 0
    TTS4 = 0
    TTSW3 = 0
    TTS1old = 0
    TTS2old = 0
    TTS3old = 0
    TTS4old = 0
    TTSW3old = 0

    numLanes = 2
#     vehPosition = []
    VSL_beforeA1 = np.round(np.array([120, 120])*(1/(3.6)),2)
    VSL_beforeA2 = np.round(np.array([120, 120])*(1/(3.6)),2)
    
    if vsl_on_off==1:
        actions_mps = np.round((np.array([60, 60, 80, 80, 100, 100, 120, 120]))*(1/3.6),2)
        position = np.array([1,3,1,3,1,3,1,3])
        A1.actions_mps = np.array([actions_mps, position])
        A2.actions_mps = np.array([actions_mps, position])
        actions_kmph = np.array([60, 60, 80, 80, 100, 100, 120, 120])
        position = np.array([1,3,1,3,1,3,1,3])
        A1.actions_kmph = np.array([actions_kmph, position])
        A2.actions_kmph = np.array([actions_kmph, position])
    else:
        actions_mps = np.round((np.array([120, 120, 120, 120, 120, 120, 120, 120]))*(1/3.6),2)
        position = np.array([1,3,1,3,1,3,1,3])
        A1.actions_mps = np.array([actions_mps, position])
        A2.actions_mps = np.array([actions_mps, position])
        actions_kmph = np.array([120, 120, 120, 120, 120, 120, 120, 120])
        position = np.array([1,3,1,3,1,3,1,3])
        A1.actions_kmph = np.array([actions_kmph, position])
        A2.actions_kmph = np.array([actions_kmph, position])

    
    #action_index_min = 0
    action_index_max_A1 = (A1.actions_mps[0,:].size)-1
    action_index_max_A2 = (A2.actions_mps[0,:].size)-1
    
    A1.prevActionWin = action_index_max_A1
    A2.prevActionWin = action_index_max_A2
    
    A1.alpha_LPi_ak = np.array([0.0, 0.0])
    A2.alpha_LPi_ak = np.array([0.0, 0.0])
    A1.alpha_LPi_ai = np.array([0.0, 0.0])
    A2.alpha_LPi_ai = np.array([0.0, 0.0])
    A1.alpha_W_LPi = np.array([0.0, 0.0])
    A2.alpha_W_LPi = np.array([0.0, 0.0])
    A1.alter_LPi_W = np.array([0.0, 0.0])
    A2.alter_LPi_W = np.array([0.0, 0.0])
    
    A1.alpha_RPi_ak = np.array([0.0, 0.0])
    A2.alpha_RPi_ak = np.array([0.0, 0.0])
    A1.alpha_RPi_ai = np.array([0.0, 0.0])
    A2.alpha_RPi_ai = np.array([0.0, 0.0])
    A1.alpha_W_RPi = np.array([0.0, 0.0])
    A2.alpha_W_RPi = np.array([0.0, 0.0])
    A1.alter_RPi_W = np.array([0.0, 0.0])
    A2.alter_RPi_W = np.array([0.0, 0.0])
    
                
    A1.actionSub_y=[np.array([0]), np.array([0])]
    A2.actionSub_y=[np.array([0]), np.array([0])]
    
    A1.x_LPi=np.array([0, 0])
    A2.x_LPi=np.array([0, 0])
    A1.y_LPi=np.array([0, 0])
    A2.y_LPi=np.array([0, 0])
    A1.R=np.array([0.0, 0.0])
    A2.R=np.array([0.0, 0.0])


    i = 0
    vf = 41.6  # 150 km/h
    j = 0

    C = 0.75  # cooperation coefficient (see our papers)
    gama = 0.8
    alpha=0.75
    beta=1.25
    w = 1.5 # parameter which controls how fast W converges (see papers)
    #epsilon = 0.8    

    # run (update) current simulation (duration 5400 s = 1.5 h)
    while(step <= 5400):
        traci.simulationStep()
    
        if step % time_sample_TTSreward == 0 and step>0:

            rewardNumVeh_L1, rewardNumVeh_L2, rewardNumVeh_L3, rewardNumVeh_L4, rewardNumVeh_W3 = rewardTTS()
            TTS1 += rewardNumVeh_L1 *(time_sample_TTSreward/3600)
            TTS2 += rewardNumVeh_L2 *(time_sample_TTSreward/3600)
            TTS3 += rewardNumVeh_L3 *(time_sample_TTSreward/3600)
            TTS4 += rewardNumVeh_L4 *(time_sample_TTSreward/3600)
            TTSW3 += rewardNumVeh_W3 *(time_sample_TTSreward/3600)
#             TTS += TTS1 + TTS2 + TTS3 + TTSW3
            
        if step % time_res_sec == 0 and step>0:
            j+=1

            numVehTotal = TTS_Total()
            TTS_Sim += numVehTotal*(time_res_sec/3600)
            
            speedL1, speedL2, speedL3, speedL4 = currentSpeed_L1L2L3L4()
            speedCellL1[j] = speedL1
            speedCellL2[j] = speedL2
            speedCellL3[j] = speedL3
            speedCellL4[j] = speedL4
            

            densityL1, densityL2, densityL3, densityL4 = current_Density_L1L2L3L4(numLanes)
            densityL1Steps[j] = densityL1
            densityL2Steps[j] = densityL2
            densityL3Steps[j] = densityL3
            densityL4Steps[j] = densityL4
            
            if plotHeatMap==1:
                speedList = speedPerSegments()
                arr = np.array(speedList)
                speedHeatMap=np.vstack([speedHeatMap,arr])            
            
        
            if step>0 and step%control_time_perid==0 and i==0:

#=============== Ai Define X=state()
                A1.x_LPi[0] = returnStateMomentum(A1.prevActionWin,\
                                                  speedCellL3[j], densityL2Steps[j], densityL3Steps[j])
    
                A1.x_LPi[1] = returnStateMomentum(A1.prevActionWin,\
                                                  speedCellL2[j], densityL2Steps[j], densityL3Steps[j])
        
                A2.x_LPi[0] = returnStateMomentum(A2.prevActionWin,\
                                              speedCellL4[j], densityL3Steps[j], densityL4Steps[j])

                A2.x_LPi[1] = returnStateMomentum(A2.prevActionWin,\
                                              speedCellL3[j], densityL3Steps[j], densityL4Steps[j])
                
                                    
#=============== A1 Define action                                        

                actionIndexA1_LP1, actionControlA1_LP1 = A1.suggestAction(epsilon, A1.x_LPi[0], A1.Q_LPi[0],\
                                                A1.prevActionWin, A1.actions_kmph[0,:], run, numberSaveData)
                actionIndexA1_LP2, actionControlA1_LP2 = A1.suggestAction(epsilon, A1.x_LPi[1], A1.Q_LPi[1],\
                                                A1.prevActionWin, A1.actions_kmph[0,:], run, numberSaveData)
                actionIndexA1_RP1, actionControlA1_RP1 = A1.suggestAction(epsilon, A2.x_LPi[0], A1.Q_RPi[0],\
                                                A1.prevActionWin, A1.actions_kmph[0,:], run, numberSaveData)
                actionIndexA1_RP2, actionControlA1_RP2 = A1.suggestAction(epsilon, A2.x_LPi[1], A1.Q_RPi[1],\
                                                A1.prevActionWin, A1.actions_kmph[0,:], run, numberSaveData)
        
#                 agent = "A1"
                A1LP1want, A1LP2want, policyA1LP_win, A1RP1want, A1RP2want,\
                remotePolicyA1RP_win, actionWinA1, policyWinA1 = actionWinner(A1.name, C,\
                                                                              A1.x_LPi[0], A1.x_LPi[1], A2.x_LPi[0],\
                                                                              A2.x_LPi[1], epsilon, actionIndexA1_LP1,\
                                                                              actionIndexA1_LP2, actionIndexA1_RP1,\
                                                                              actionIndexA1_RP2, A1.W_LPi[0],\
                                                                              A1.W_LPi[1], A1.W_RPi[0], A1.W_RPi[1]) 
    
                A1.actionWin = actionWinA1
                a1 = A1.actions_mps[0, actionWinA1]
                A1.VSLposition = A1.actions_mps[1, actionWinA1]
#                 A1.prevActionWin = A1.actionWin
                A1.actionWant_LPi = [actionIndexA1_LP1, actionIndexA1_LP2]
                A1.actionWant_RPi = [actionIndexA1_RP1, actionIndexA1_RP2]
                A1.actionControl_LPi = [actionControlA1_LP1, actionControlA1_LP2]
                A1.actionControl_RPi = [actionControlA1_RP1, actionControlA1_RP2]
                A1.LPiWin = policyA1LP_win
                A1.RPiWin = remotePolicyA1RP_win
                A1.policyWin = policyWinA1

#=============== A1 END of Define action


#================ A2 Define action

                actionIndexA2_LP1, actionControlA2_LP1 = A2.suggestAction(epsilon, A2.x_LPi[0], A2.Q_LPi[0],\
                                                A2.prevActionWin, A2.actions_kmph[0,:], run, numberSaveData)
                actionIndexA2_LP2, actionControlA2_LP2 = A2.suggestAction(epsilon, A2.x_LPi[1], A2.Q_LPi[1],\
                                                A2.prevActionWin, A2.actions_kmph[0,:], run, numberSaveData)
        
                actionIndexA2_RP1, actionControlA2_RP1 = A2.suggestAction(epsilon, A1.x_LPi[0], A2.Q_RPi[0],\
                                                A2.prevActionWin, A2.actions_kmph[0,:], run, numberSaveData)
                actionIndexA2_RP2, actionControlA2_RP2 = A2.suggestAction(epsilon, A1.x_LPi[1], A2.Q_RPi[1],\
                                                A2.prevActionWin, A2.actions_kmph[0,:], run, numberSaveData)
#                 agent = "A1"
                A2LP1want, A2LP2want, policyA2LP_win, A2RP1want, A2RP2want,\
                remotePolicyA2RP_win, actionWinA2, policyWinA2 = actionWinner(A2.name, C, A2.x_LPi[0], A2.x_LPi[1],\
                                                                              A1.x_LPi[0], A1.x_LPi[1], epsilon,\
                                                                              actionIndexA2_LP1, actionIndexA2_LP2,\
                                                                              actionIndexA2_RP1, actionIndexA2_RP2,\
                                                                              A2.W_LPi[0], A2.W_LPi[1], A2.W_RPi[0],\
                                                                              A2.W_RPi[1])
    
                A2.actionWin = actionWinA2
                a2 = A2.actions_mps[0, actionWinA2]
                A2.VSLposition = A2.actions_mps[1, actionWinA2]
#                 A2.prevActionWin = A2.actionWin
                A2.actionWant_LPi = [actionIndexA2_LP1, actionIndexA2_LP2]
                A2.actionWant_RPi = [actionIndexA2_RP1, actionIndexA2_RP2]
                A2.actionControl_LPi = [actionControlA2_LP1, actionControlA2_LP2]
                A2.actionControl_RPi = [actionControlA2_RP1, actionControlA2_RP2]
                A2.LPiWin = policyA2LP_win
                A2.RPiWin = remotePolicyA2RP_win
                A2.policyWin = policyWinA2
                
#===============A2 END of Define action

                A1.prevActionWin = A1.actionWin    
                A2.prevActionWin = A2.actionWin


#                 occupancy = occupancyOnRamp()
                VSL_beforeA1[1]=a1
                VSL_beforeA2[1]=a2
            
                vehIDs_A1, vehIDs_A2, sF_org_L1_list, sF_org_L2_list, speedLimitList = setSpeedLimit(a1,a2,\
                                                                                            A1.VSLposition,\
                                                                                             A2.VSLposition,\
                                                                                     VSL_beforeA1,VSL_beforeA2,\
                                                                                    plotHeatMap, speedLimitList)
                
                arra = np.array(speedLimitList)
                speedLimitHeatMap=np.vstack([speedLimitHeatMap,arra])                
            
                VSL_beforeA1[0] = VSL_beforeA1[1]
                VSL_beforeA2[0] = VSL_beforeA2[1]
               
                speedCellL1[0] = speedCellL1[j]
                speedCellL1[1:] = 0
                speedCellL2[0] = speedCellL2[j]
                speedCellL2[1:] = 0
                speedCellL3[0] = speedCellL3[j]
                speedCellL3[1:] = 0
                speedCellL4[0] = speedCellL4[j]
                speedCellL4[1:] = 0
                
                densityL1Steps[0] = densityL1Steps[j]
                densityL1Steps[1:] = 0
                densityL2Steps[0] = densityL2Steps[j]
                densityL2Steps[1:] = 0
                densityL3Steps[0] = densityL3Steps[j]
                densityL3Steps[1:] = 0
                densityL4Steps[0] = densityL4Steps[j]
                densityL4Steps[1:] = 0
                
                TTS1 = 0
                TTS2 = 0
                TTS3 = 0
                TTS4 = 0
                TTSW3 = 0
                TTS = 0
                
                i=1
                j=0

                
            elif step>0 and step%control_time_perid==0 and i==1:
                #=================================================================
                 # A1 policies rewards               
                if speedCellL2[j]>28.33 and speedCellL3[j]>28.33:
                    A1.R[0] = 0
#                     r2 = 0
                else:               
                    A1.R[0]=-TTS2
                
                A1.R[1]=-alpha*TTS3
                #==================================================================
                # A2 policies rewards
                if speedCellL3[j]>28.33 and speedCellL4[j]>28.33:
                    A2.R[0] = 0
#                     r2 = 0
                else:               
                    A2.R[0]=-TTS3
                
                A2.R[1]=-beta*TTS4
                #==================================================================         
                
                A1.y_LPi[0] = returnStateMomentum(A1.prevActionWin,\
                                                  speedCellL3[j], densityL2Steps[j], densityL3Steps[j])
    
                A1.y_LPi[1] = returnStateMomentum(A1.prevActionWin,\
                                                  speedCellL2[j], densityL2Steps[j], densityL3Steps[j])
        
                A2.y_LPi[0] = returnStateMomentum(A2.prevActionWin,\
                                              speedCellL4[j], densityL3Steps[j], densityL4Steps[j])

                A2.y_LPi[1] = returnStateMomentum(A2.prevActionWin,\
                                              speedCellL3[j], densityL3Steps[j], densityL4Steps[j])

                

                occupancy = occupancyOnRamp()
                
                if update_QW == 1:

    #=================Update Local Policies (LP) for Ai
                    for idAi, Ai in enumerate(Agent):
    #=================Update Ai's LPi
                        for indx, Q_LP in enumerate(Ai.Q_LPi):
                            action_subset_y_Ai_LPi = Ai.subsetAction_in_y(Ai.actionWin, Ai.actions_kmph[0,:],\
                                                                          run, numberSaveData)
                            Ai.actionSub_y[indx]=action_subset_y_Ai_LPi

                            alphaQ_AiLPi_ak = Ai.alpha_Q_ak_Function(Ai.x_LPi[indx], Ai.actionWin,\
                                                                     Ai.Num_Visited_LPi_x_ak[indx], 1)                  

                            Ai.updateQ(Ai.x_LPi[indx], Ai.actionWin, Ai.y_LPi[indx], action_subset_y_Ai_LPi,\
                                       Ai.R[indx], alphaQ_AiLPi_ak, gama, Ai.Q_LPi[indx])


                        if Ai.policyWin=="RP1" or Ai.policyWin=="RP2":
                            alphaQ_Ai_LP1_ai = Ai.alpha_Q_ak_Function(Ai.x_LPi[0], Ai.actionWant_LPi[0],\
                                                  Ai.Num_Visited_LPi_x_ak[0], 0)
                            alphaAi_W_LP1 = Ai.alpha_W_Function(Ai.x_LPi[0], Ai.Num_Visited_LPi_x[0], 1)
                            Ai.updateW(Ai.x_LPi[0], Ai.actionWant_LPi[0], Ai.y_LPi[0], Ai.actionSub_y[0],\
                                       Ai.R[0], alphaAi_W_LP1, alphaQ_Ai_LP1_ai, gama, w, Ai.W_LPi[0], Ai.Q_LPi[0])
                            alterAi_LP1_W = alphaAi_W_LP1*(1-alphaQ_Ai_LP1_ai)**w

                            alphaQ_Ai_LP2_ai = Ai.alpha_Q_ak_Function(Ai.x_LPi[1], Ai.actionWant_LPi[1],\
                                                  Ai.Num_Visited_LPi_x_ak[1], 0)
                            alphaAi_W_LP2 = Ai.alpha_W_Function(Ai.x_LPi[1], Ai.Num_Visited_LPi_x[1], 1)
                            Ai.updateW(Ai.x_LPi[1], Ai.actionWant_LPi[1], Ai.y_LPi[1], Ai.actionSub_y[1],\
                                       Ai.R[1], alphaAi_W_LP2, alphaQ_Ai_LP2_ai, gama, w, Ai.W_LPi[1], Ai.Q_LPi[1])
                            alterAi_LP2_W = alphaAi_W_LP2*(1-alphaQ_Ai_LP2_ai)**w        
                        else:
                            if Ai.LPiWin == "LP2":
                                alphaQ_Ai_LP1_ai = Ai.alpha_Q_ak_Function(Ai.x_LPi[0], Ai.actionWant_LPi[0],\
                                                      Ai.Num_Visited_LPi_x_ak[0], 0)
                                alphaAi_W_LP1 = Ai.alpha_W_Function(Ai.x_LPi[0], Ai.Num_Visited_LPi_x[0], 1)
                                Ai.updateW(Ai.x_LPi[0], Ai.actionWant_LPi[0], Ai.y_LPi[0], Ai.actionSub_y[0],\
                                           Ai.R[0], alphaAi_W_LP1, alphaQ_Ai_LP1_ai, gama, w, Ai.W_LPi[0], Ai.Q_LPi[0])
                                alterAi_LP1_W = alphaAi_W_LP1*(1-alphaQ_Ai_LP1_ai)**w
                            else:
                                alphaQ_Ai_LP2_ai = Ai.alpha_Q_ak_Function(Ai.x_LPi[1], Ai.actionWant_LPi[1],\
                                                      Ai.Num_Visited_LPi_x_ak[1], 0)
                                alphaAi_W_LP2 = Ai.alpha_W_Function(Ai.x_LPi[1], Ai.Num_Visited_LPi_x[1], 1)
                                Ai.updateW(Ai.x_LPi[1], Ai.actionWant_LPi[1], Ai.y_LPi[1], Ai.actionSub_y[1],\
                                           Ai.R[1], alphaAi_W_LP2, alphaQ_Ai_LP2_ai, gama, w, Ai.W_LPi[1], Ai.Q_LPi[1])
                                alterAi_LP2_W = alphaAi_W_LP2*(1-alphaQ_Ai_LP2_ai)**w
                            
                        # used for analytics, saved in controlFile
                        Ai.alpha_LPi_ak[0] = Ai.alpha_Q_ak_Function(Ai.x_LPi[0], Ai.actionWin,\
                                                                     Ai.Num_Visited_LPi_x_ak[0], 0)
                        Ai.alpha_LPi_ak[1] = Ai.alpha_Q_ak_Function(Ai.x_LPi[1], Ai.actionWin,\
                                             Ai.Num_Visited_LPi_x_ak[1], 0)
                        
                        Ai.alpha_LPi_ai[0] = Ai.alpha_Q_ak_Function(Ai.x_LPi[0], Ai.actionWant_LPi[0],\
                                                  Ai.Num_Visited_LPi_x_ak[0], 0)
                        Ai.alpha_LPi_ai[1] = Ai.alpha_Q_ak_Function(Ai.x_LPi[1], Ai.actionWant_LPi[1],\
                                                  Ai.Num_Visited_LPi_x_ak[1], 0)

                        Ai.alpha_W_LPi[0] = Ai.alpha_W_Function(Ai.x_LPi[0], Ai.Num_Visited_LPi_x[0], 0)
                        Ai.alpha_W_LPi[1] = Ai.alpha_W_Function(Ai.x_LPi[1], Ai.Num_Visited_LPi_x[1], 0)

                        Ai.alter_LPi_W[0] = Ai.alpha_W_LPi[0]*(1-Ai.alpha_LPi_ai[0])**w
                        Ai.alter_LPi_W[1] = Ai.alpha_W_LPi[1]*(1-Ai.alpha_LPi_ai[1])**w

    #================== End update LP

    
    
    #================== Update Remote Policies (RP) for Ai
                    agentList=[]
                    for Ai in Agent:
                        agentList.append(Ai)
                    for idAi, Ai in enumerate(Agent):
                        LPij=1-idAi
    #=================Update Ai's RPi
                        for indxR, Q_RP in enumerate(Ai.Q_RPi):
                            alphaQR_AiRPi_ak = Ai.alpha_Q_ak_Function(agentList[LPij].x_LPi[indxR], Ai.actionWin,\
                                                                      Ai.Num_Visited_RPi_x_ak[indxR], 1)

                            Ai.updateQ(agentList[LPij].x_LPi[indxR], Ai.actionWin, agentList[LPij].y_LPi[indxR],\
                                       Ai.actionSub_y[indxR], agentList[LPij].R[indxR], alphaQR_AiRPi_ak, gama, Ai.Q_RPi[indxR])

                        if Ai.policyWin=="LP1" or Ai.policyWin=="LP2":
                            alphaQR_Ai_RP1_ai = Ai.alpha_Q_ak_Function(agentList[LPij].x_LPi[0],  Ai.actionWant_RPi[0],\
                                                               Ai.Num_Visited_RPi_x_ak[0], 0)
                            alphaAi_W_RP1 = Ai.alpha_W_Function(agentList[LPij].x_LPi[0], Ai.Num_Visited_RPi_x[0], 1)
                            Ai.updateW(agentList[LPij].x_LPi[0], Ai.actionWant_RPi[0], agentList[LPij].y_LPi[0],\
                                       Ai.actionSub_y[0], agentList[LPij].R[0], alphaAi_W_RP1, alphaQR_Ai_RP1_ai,\
                                       gama, w, Ai.W_RPi[0], Ai.Q_RPi[0])
                            alterAi_RP1_W = alphaAi_W_RP1*(1-alphaQR_Ai_RP1_ai)**w

                            alphaQR_Ai_RP2_ai = Ai.alpha_Q_ak_Function(agentList[LPij].x_LPi[1],  Ai.actionWant_RPi[1],\
                                       Ai.Num_Visited_RPi_x_ak[1], 0)
                            alphaAi_W_RP2 = Ai.alpha_W_Function(agentList[LPij].x_LPi[1], Ai.Num_Visited_RPi_x[1], 1)
                            Ai.updateW(agentList[LPij].x_LPi[1], Ai.actionWant_RPi[1], agentList[LPij].y_LPi[1],\
                                       Ai.actionSub_y[1], agentList[LPij].R[1], alphaAi_W_RP2, alphaQR_Ai_RP2_ai,\
                                       gama, w, Ai.W_RPi[1], Ai.Q_RPi[1])
                            alterAi_RP2_W = alphaAi_W_RP2*(1-alphaQR_Ai_RP2_ai)**w
                
                        else:
                            if Ai.RPiWin == "RP2":
                                alphaQR_Ai_RP1_ai = Ai.alpha_Q_ak_Function(agentList[LPij].x_LPi[0],  Ai.actionWant_RPi[0],\
                                                                   Ai.Num_Visited_RPi_x_ak[0], 0)
                                alphaAi_W_RP1 = Ai.alpha_W_Function(agentList[LPij].x_LPi[0], Ai.Num_Visited_RPi_x[0], 1)
                                Ai.updateW(agentList[LPij].x_LPi[0], Ai.actionWant_RPi[0], agentList[LPij].y_LPi[0],\
                                           Ai.actionSub_y[0], agentList[LPij].R[0], alphaAi_W_RP1, alphaQR_Ai_RP1_ai,\
                                           gama, w, Ai.W_RPi[0], Ai.Q_RPi[0])
                                alterAi_RP1_W = alphaAi_W_RP1*(1-alphaQR_Ai_RP1_ai)**w
                            else:
                                alphaQR_Ai_RP2_ai = Ai.alpha_Q_ak_Function(agentList[LPij].x_LPi[1],  Ai.actionWant_RPi[1],\
                                           Ai.Num_Visited_RPi_x_ak[1], 0)
                                alphaAi_W_RP2 = Ai.alpha_W_Function(agentList[LPij].x_LPi[1], Ai.Num_Visited_RPi_x[1], 1)
                                Ai.updateW(agentList[LPij].x_LPi[1], Ai.actionWant_RPi[1], agentList[LPij].y_LPi[1],\
                                           Ai.actionSub_y[1], agentList[LPij].R[1], alphaAi_W_RP2, alphaQR_Ai_RP2_ai,\
                                           gama, w, Ai.W_RPi[1], Ai.Q_RPi[1])
                                alterAi_RP2_W = alphaAi_W_RP2*(1-alphaQR_Ai_RP2_ai)**w
                            
                        # used for analytics, saved in controlFile
                        Ai.alpha_RPi_ak[0] = Ai.alpha_Q_ak_Function(agentList[LPij].x_LPi[0], Ai.actionWin,\
                                          Ai.Num_Visited_RPi_x_ak[0], 0)
                        Ai.alpha_RPi_ak[1] = Ai.alpha_Q_ak_Function(agentList[LPij].x_LPi[1], Ai.actionWin,\
                                          Ai.Num_Visited_RPi_x_ak[1], 0)
                        
                        Ai.alpha_RPi_ai[0] = Ai.alpha_Q_ak_Function(agentList[LPij].x_LPi[0],\
                                                                    Ai.actionWant_RPi[0], Ai.Num_Visited_RPi_x_ak[0], 0)
                        Ai.alpha_RPi_ai[1] = Ai.alpha_Q_ak_Function(agentList[LPij].x_LPi[1],\
                                                                    Ai.actionWant_RPi[1], Ai.Num_Visited_RPi_x_ak[1], 0)

                        Ai.alpha_W_RPi[0] = Ai.alpha_W_Function(agentList[LPij].x_LPi[0], Ai.Num_Visited_RPi_x[0], 0)
                        Ai.alpha_W_RPi[1] = Ai.alpha_W_Function(agentList[LPij].x_LPi[1], Ai.Num_Visited_RPi_x[1], 0)

                        Ai.alter_RPi_W[0] = Ai.alpha_W_RPi[0]*(1-Ai.alpha_RPi_ai[0])**w
                        Ai.alter_RPi_W[1] = Ai.alpha_W_RPi[1]*(1-Ai.alpha_RPi_ai[1])**w

                
                TTS1old = TTS1
                TTS2old = TTS2
                TTS3old = TTS3
                TTS4old = TTS4
                TTSW3old = TTSW3
                
                TTS1 = 0
                TTS2 = 0
                TTS3 = 0
                TTS4 = 0
                TTSW3 = 0
            
                A1.x_old_LPi = deepcopy(A1.x_LPi) # = xA1_LP2 
                A2.x_old_LPi = deepcopy(A2.x_LPi) # = xA2_LP2
                # X' --> X, and select new action a
                A1.x_LPi=deepcopy(A1.y_LPi)
                A2.x_LPi=deepcopy(A2.y_LPi)
                

#=============== A1 Define action                                        

                actionIndexA1_LP1, actionControlA1_LP1 = A1.suggestAction(epsilon, A1.x_LPi[0], A1.Q_LPi[0],\
                                                A1.prevActionWin, A1.actions_kmph[0,:], run, numberSaveData)
                actionIndexA1_LP2, actionControlA1_LP2 = A1.suggestAction(epsilon, A1.x_LPi[1], A1.Q_LPi[1],\
                                                A1.prevActionWin, A1.actions_kmph[0,:], run, numberSaveData)
                actionIndexA1_RP1, actionControlA1_RP1 = A1.suggestAction(epsilon, A2.x_LPi[0], A1.Q_RPi[0],\
                                                A1.prevActionWin, A1.actions_kmph[0,:], run, numberSaveData)
                actionIndexA1_RP2, actionControlA1_RP2 = A1.suggestAction(epsilon, A2.x_LPi[1], A1.Q_RPi[1],\
                                                A1.prevActionWin, A1.actions_kmph[0,:], run, numberSaveData)
        
#                 agent = "A1"
                A1LP1want, A1LP2want, policyA1LP_win, A1RP1want, A1RP2want,\
                remotePolicyA1RP_win, actionWinA1, policyWinA1 = actionWinner(A1.name, C,\
                                                                              A1.x_LPi[0], A1.x_LPi[1], A2.x_LPi[0],\
                                                                              A2.x_LPi[1], epsilon, actionIndexA1_LP1,\
                                                                              actionIndexA1_LP2, actionIndexA1_RP1,\
                                                                              actionIndexA1_RP2, A1.W_LPi[0],\
                                                                              A1.W_LPi[1], A1.W_RPi[0], A1.W_RPi[1]) 
                A1.actionWin = actionWinA1
                a1 = A1.actions_mps[0, actionWinA1]
                A1.VSLposition = A1.actions_mps[1, actionWinA1]
#                 A1.prevActionWin = A1.actionWin
                A1.actionWant_LPi = [actionIndexA1_LP1, actionIndexA1_LP2]
                A1.actionWant_RPi = [actionIndexA1_RP1, actionIndexA1_RP2]
                A1.actionControl_LPi = [actionControlA1_LP1, actionControlA1_LP2]
                A1.actionControl_RPi = [actionControlA1_RP1, actionControlA1_RP2]
                A1.LPiWin = policyA1LP_win
                A1.RPiWin = remotePolicyA1RP_win
                A1.policyWin = policyWinA1
                
#=============== A1 END of Define action


#================ A2 Define action

                actionIndexA2_LP1, actionControlA2_LP1 = A2.suggestAction(epsilon, A2.x_LPi[0], A2.Q_LPi[0],\
                                                A2.prevActionWin, A2.actions_kmph[0,:], run, numberSaveData)
                actionIndexA2_LP2, actionControlA2_LP2 = A2.suggestAction(epsilon, A2.x_LPi[1], A2.Q_LPi[1],\
                                                A2.prevActionWin, A2.actions_kmph[0,:], run, numberSaveData)
        
                actionIndexA2_RP1, actionControlA2_RP1 = A2.suggestAction(epsilon, A1.x_LPi[0], A2.Q_RPi[0],\
                                        A2.prevActionWin, A2.actions_kmph[0,:], run, numberSaveData)
                actionIndexA2_RP2, actionControlA2_RP2 = A2.suggestAction(epsilon, A1.x_LPi[1], A2.Q_RPi[1],\
                                                A2.prevActionWin, A2.actions_kmph[0,:], run, numberSaveData)
#                 agent = "A1"
                A2LP1want, A2LP2want, policyA2LP_win, A2RP1want, A2RP2want,\
                remotePolicyA2RP_win, actionWinA2, policyWinA2 = actionWinner(A2.name, C, A2.x_LPi[0], A2.x_LPi[1],\
                                                                              A1.x_LPi[0], A1.x_LPi[1], epsilon,\
                                                                              actionIndexA2_LP1, actionIndexA2_LP2,\
                                                                              actionIndexA2_RP1, actionIndexA2_RP2,\
                                                                              A2.W_LPi[0], A2.W_LPi[1], A2.W_RPi[0],\
                                                                              A2.W_RPi[1])
                A2.actionWin = actionWinA2
                a2 = A2.actions_mps[0, actionWinA2]
                A2.VSLposition = A2.actions_mps[1, actionWinA2]
#                 A2.prevActionWin = A2.actionWin
                A2.actionWant_LPi = [actionIndexA2_LP1, actionIndexA2_LP2]
                A2.actionWant_RPi = [actionIndexA2_RP1, actionIndexA2_RP2]
                A2.actionControl_LPi = [actionControlA2_LP1, actionControlA2_LP2]
                A2.actionControl_RPi = [actionControlA2_RP1, actionControlA2_RP2]
                A2.LPiWin = policyA2LP_win
                A2.RPiWin = remotePolicyA2RP_win
                A2.policyWin = policyWinA2
                
#================ A2 END of Define action

                VSL_beforeA1[1] = a1
                VSL_beforeA2[1] = a2
        
                vehIDs_A1, vehIDs_A2, sF_org_L1_list, sF_org_L2_list, speedLimitList = setSpeedLimit(a1,a2,\
                                                                                            A1.VSLposition,\
                                                                                             A2.VSLposition,\
                                                                                     VSL_beforeA1,VSL_beforeA2,\
                                                                                    plotHeatMap, speedLimitList)
                
                
                arra = np.array(speedLimitList)
                speedLimitHeatMap=np.vstack([speedLimitHeatMap,arra])
                
                VSL_beforeA1[0] = VSL_beforeA1[1]
                VSL_beforeA2[0] = VSL_beforeA2[1]
                if run%numberSaveData==0 or run>end_learning:
                    controlFile = np.vstack([controlFile,[str(A1.x_old_LPi[0])+'-'+str(A1.x_old_LPi[1]),\
                                                          str(A2.x_old_LPi[0])+'-'+str(A2.x_old_LPi[1]),\
                                                          format(speedCellL1[0],".1f"),\
                                                          format(speedCellL2[0],".1f"),\
                                                          format(speedCellL3[0],".1f"),\
                                                          format(speedCellL4[0],".1f"),\
                                                          format(densityL1Steps[0],".1f"),\
                                                          format(densityL2Steps[0],".1f"),\
                                                          format(densityL3Steps[0],".1f"),\
                                                          format(densityL4Steps[0],".1f"),\
                                                          format(A1.actions_mps[0, A1.prevActionWin],".1f"),\
                                                          format(A2.actions_mps[0, A2.prevActionWin],".1f"),\
                                                          format(TTS1old,".2f"),\
                                                          format(TTS2old,".2f"),\
                                                          format(TTS3old,".2f"),\
                                                          format(TTS4old,".2f"),\
                                                          format(TTSW3old,".2f"),\
                                                          rewardNumVeh_W3,\
                                                          format(A1.R[0],".2f"),\
                                                          format(A1.R[1],".2f"),\
                                                          format(A2.R[0],".2f"),\
                                                          format(A2.R[1],".2f"),\
                                                          str(A1.x_LPi[0])+'-'+str(A1.x_LPi[1]),\
                                                          str(A2.x_LPi[0])+'-'+str(A2.x_LPi[1]),\
                                                          format(speedCellL1[j],".1f"),\
                                                          format(speedCellL2[j],".1f"),\
                                                          format(speedCellL3[j],".1f"),\
                                                          format(speedCellL4[j],".1f"),\
                                                          format(densityL1Steps[j],".1f"),\
                                                          format(densityL2Steps[j],".1f"),\
                                                          format(densityL3Steps[j],".1f"),\
                                                          format(densityL4Steps[j],".1f"),\
                                                          format(A1.actions_mps[0,actionWinA1],".1f"),\
                                                          A1.actionWant_LPi[0],\
                                                          A1.actionWant_LPi[1],\
                                                          A1.actionWant_RPi[0],\
                                                          A1.actionWant_RPi[1],\
                                                          A1.policyWin,\
                                                          A1.actionControl_LPi[0],\
                                                          A1.actionControl_LPi[1],\
                                                          A1.actionControl_RPi[0],\
                                                          A1.actionControl_RPi[1],\
                                                          format(A2.actions_mps[0,actionWinA2],".1f"),\
                                                          A2.actionWant_LPi[0],\
                                                          A2.actionWant_LPi[1],\
                                                          A2.actionWant_RPi[0],\
                                                          A2.actionWant_RPi[1],\
                                                          A2.policyWin,\
                                                          A2.actionControl_LPi[0],\
                                                          A2.actionControl_LPi[1],\
                                                          A2.actionControl_RPi[0],\
                                                          A2.actionControl_RPi[1],\
                                                          format(A1.W_LPi[0][A1.x_LPi[0],0],".6f"),\
                                                          format(A1.W_LPi[1][A1.x_LPi[1],0],".6f"),\
                                                          format(A1.W_RPi[0][A2.x_LPi[0],0],".6f"),\
                                                          format(A1.W_RPi[1][A2.x_LPi[1],0],".6f"),\
                                                          format(A2.W_LPi[0][A2.x_LPi[0],0],".6f"),\
                                                          format(A2.W_LPi[1][A2.x_LPi[1],0],".6f"),\
                                                          format(A2.W_RPi[0][A1.x_LPi[0],0],".6f"),\
                                                          format(A2.W_RPi[1][A1.x_LPi[1],0],".6f"),\
                                                          format(A1.alpha_LPi_ai[0],".7f"),\
                                                          format(A1.alpha_LPi_ai[1],".7f"),\
                                                          format(A1.alpha_RPi_ai[0],".7f"),\
                                                          format(A1.alpha_RPi_ai[1],".7f"),\
                                                          format(A2.alpha_LPi_ai[0],".7f"),\
                                                          format(A2.alpha_LPi_ai[1],".7f"),\
                                                          format(A2.alpha_RPi_ai[0],".7f"),\
                                                          format(A2.alpha_RPi_ai[1],".7f"),\
                                                          format(A1.alpha_LPi_ak[0],".7f"),\
                                                          format(A1.alpha_LPi_ak[1],".7f"),\
                                                          format(A1.alpha_RPi_ak[0],".7f"),\
                                                          format(A1.alpha_RPi_ak[1],".7f"),\
                                                          format(A2.alpha_LPi_ak[0],".7f"),\
                                                          format(A2.alpha_LPi_ak[1],".7f"),\
                                                          format(A2.alpha_RPi_ak[0],".7f"),\
                                                          format(A2.alpha_RPi_ak[1],".7f"),\
                                                          format(A1.alpha_W_LPi[0],".7f"),\
                                                          format(A1.alpha_W_LPi[1],".7f"),\
                                                          format(A1.alpha_W_RPi[0],".7f"),\
                                                          format(A1.alpha_W_RPi[1],".7f"),\
                                                          format(A2.alpha_W_LPi[0],".7f"),\
                                                          format(A2.alpha_W_LPi[1],".7f"),\
                                                          format(A2.alpha_W_RPi[0],".7f"),\
                                                          format(A2.alpha_W_RPi[1],".7f"),\
                                                          format(A1.alter_LPi_W[0],".7f"),\
                                                          format(A1.alter_LPi_W[1],".7f"),\
                                                          format(A1.alter_RPi_W[0],".7f"),\
                                                          format(A1.alter_RPi_W[1],".7f"),\
                                                          format(A2.alter_LPi_W[0],".7f"),\
                                                          format(A2.alter_LPi_W[1],".7f"),\
                                                          format(A2.alter_RPi_W[0],".7f"),\
                                                          format(A2.alter_RPi_W[1],".7f"),\
                                                          format(TTS_Sim,".2f"),\
                                                          format(occupancy[0],".4f"),\
                                                          format(occupancy[1],".4f"),\
                                                          format(occupancy[2],".4f"),\
                                                          A1.actionWin,\
                                                          A1.LPiWin,\
                                                          A1.RPiWin,\
                                                          A2.actionWin,\
                                                          A2.LPiWin,\
                                                          A2.RPiWin,\
                                                          A1.VSLposition,\
                                                          A2.VSLposition]])


 
                A1.prevActionWin = deepcopy(A1.actionWin)     
                A2.prevActionWin = deepcopy(A2.actionWin)

                # remove the current "last" time step state as the first element in new cycles
                # we measure states each 50 [s] thus 4 times per control time step 150 [s]
                # however, we use always last "current" measurements as relevant for states (one can play and try to use it average)
                speedCellL1[0] = speedCellL1[j]
                speedCellL1[1:] = 0
                speedCellL2[0] = speedCellL2[j]
                speedCellL2[1:] = 0
                speedCellL3[0] = speedCellL3[j]
                speedCellL3[1:] = 0
                speedCellL4[0] = speedCellL4[j]
                speedCellL4[1:] = 0
                
                densityL1Steps[0] = densityL1Steps[j]
                densityL1Steps[1:] = 0
                densityL2Steps[0] = densityL2Steps[j]
                densityL2Steps[1:] = 0
                densityL3Steps[0] = densityL3Steps[j]
                densityL3Steps[1:] = 0
                densityL4Steps[0] = densityL4Steps[j]
                densityL4Steps[1:] = 0
                
                j = 0


                

                             
        if step>control_time_perid and len(set(vehIDs_A1))!=0 and step%1==0:
            agent="A1"
            VSL1pos = A1.actions_mps[1, actionWinA1]
            setSpeedFactorL1(vehIDs_A1, agent, sF_org_L1_list, VSL1pos)
            
                             
        if step>control_time_perid and len(set(vehIDs_A2))!=0 and step%1==0:
            agent="A2"
            VSL2pos = A2.actions_mps[1, actionWinA2]
            setSpeedFactorL2(vehIDs_A2, agent, sF_org_L2_list, VSL2pos)
        
        step += 1
        
    saveData(vsl_on_off, run, numberSaveData, end_learning)
    
    
    traci.close()
    run+=1  
