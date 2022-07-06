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


# PLOT BEST SCENARIO WITH VSL VS. NO-VSL
#=======================================
#=======================================

# ================== START Analytics 
# (search for best trained scenario e.g., regarding minimization of Total Travel Time (Spent) - TTS)

from scipy.ndimage import gaussian_filter
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path_results = "C:/Users/kusic/Desktop/KK/SUMO/WL_analysis/reward_freeSpeed_or_TTS/"+""+\
"DWL_v2p1p2_betaPonder_1TTS3/GIT/MD_C_0_75_W_on_LP_RP/Results/"

start_exploit=14000+1 # Start exploitation phase
end_exploit=14100 # End exploitation phase

df_TTS=pd.DataFrame(columns=['TTS','indx'])
df_Speed=pd.DataFrame(columns=['V4(t)','indx'])
df_Density=pd.DataFrame(columns=['denL4(t)','indx'])

       
for i in range(start_exploit,end_exploit,1):
    data = pd.read_excel(path_results+"controlFile"+str(i)+".xlsx")
    df_TTS = pd.DataFrame(np.vstack((df_TTS, [data.loc[data.index[-1], "TTSsim"],i])))
    df_Speed = pd.DataFrame(np.vstack((df_Speed, [data["V4(t)"].mean(),i])))
    df_Density = pd.DataFrame(np.vstack((df_Density, [data["denL4(t)"].mean(),i])))
        
df_TTS.columns=['TTS','indx']
print('TTS_min',df_TTS[['TTS']].min())
df_TTS['TTS'] = pd.to_numeric(df_TTS['TTS'])
print('TTS_min_arg', df_TTS.loc[df_TTS[['TTS']].idxmin(), 'indx'].item())

df_Speed.columns=['V4(t)','indx']
print('Speed_max',df_Speed[['V4(t)']].max())
df_Speed['V4(t)'] = pd.to_numeric(df_Speed['V4(t)'])
print('Speed_max_arg', df_Speed.loc[df_Speed[['V4(t)']].idxmin(), 'indx'].item())

df_Density.columns=['denL4(t)','indx']
print('Density_min',df_Density[['denL4(t)']].min())
df_Density['denL4(t)'] = pd.to_numeric(df_Density['denL4(t)'])
print('Density_min_arg', df_Density.loc[df_Density[['denL4(t)']].idxmin(), 'indx'].item())

indx_best = int(df_TTS.loc[df_TTS[['TTS']].idxmin(), 'indx'].item()) # one can play with different objectives (e.g., min Density, or max Speed)
print("Best solution min(TTS) saved in controlFile",indx_best)
# ================== END Analytics

# NO-VSL
df_TTS_NO_VSL=pd.DataFrame(columns=['TTS','indx'])
df_Speed_NO_VSL=pd.DataFrame(columns=['V4(t)','indx'])
df_Density_NO_VSL=pd.DataFrame(columns=['denL4(t)','indx'])

data = pd.read_excel(path_results+"NO_VSL_controlFile"+str(end_exploit+1)+".xlsx")
df_TTS_NO_VSL = pd.DataFrame(np.vstack((df_TTS_NO_VSL, [data.loc[data.index[-1], "TTSsim"],i])))
df_Speed_NO_VSL = pd.DataFrame(np.vstack((df_Speed_NO_VSL, [data["V4(t)"].mean(),i])))
df_Density_NO_VSL = pd.DataFrame(np.vstack((df_Density_NO_VSL, [data["denL4(t)"].mean(),i])))

df_TTS_NO_VSL.columns=['TTS','indx']
print('TTS_NO_VSL_min',df_TTS_NO_VSL[['TTS']].min())
df_TTS_NO_VSL['TTS'] = pd.to_numeric(df_TTS_NO_VSL['TTS'])

df_Speed_NO_VSL.columns=['V4(t)','indx']
print('Speed_NO_VSL_max',df_Speed_NO_VSL[['V4(t)']].max())
df_Speed_NO_VSL['V4(t)'] = pd.to_numeric(df_Speed_NO_VSL['V4(t)'])


df_Density_NO_VSL.columns=['denL4(t)','indx']
print('Density_NO_VSL_min',df_Density_NO_VSL[['denL4(t)']].min())
df_Density_NO_VSL['denL4(t)'] = pd.to_numeric(df_Density_NO_VSL['denL4(t)'])


# PLOT BEST SCENARIO WITH VSL VS. NO-VSL
#=======================================
# ================== Plot Speed heat map for NO-VSL
heatMapMatrix = np.loadtxt(open(path_results+'NO_VSL_speedHeatMap'+str(end_exploit+1)+'.csv', 'rt'), delimiter=",")
df_for_smooth = pd.DataFrame(heatMapMatrix[1:,:]*3.6)
df = gaussian_filter(df_for_smooth, sigma=0.5)
ax = sns.heatmap(df, cmap="jet_r", vmin=0, vmax=120, cbar_kws={'label': 'Speed [km/h]'},rasterized=True)
ax.invert_yaxis()
xticks_labels = [0,1,2,3,4,5,6,7,8]#[0, 2000, 4000, 6000, 8000]
plt.xticks(np.arange(0,161,20), xticks_labels, rotation=0)
yticks_labels = [0, .25, .5, .75, 1, 1.25, 1.5]
plt.yticks(np.arange(0,109,18), yticks_labels, rotation=0)
plt.ylabel('Time [h]')
plt.xlabel('Distance [km]')
plt.title('NO-VSL (spatioltemporal flow speed)')
plt.savefig("Speed_heatMap_DWL_VSLmoving_MD"+str(end_exploit+1)+".png",format='png')
plt.savefig("Speed_heatMap_DWL_VSLmoving_MD"+str(end_exploit+1)+".pdf",format='pdf')
plt.show()
# ================== END Plot Speed heat map for NO-VSLo

# ================== Plot Speed heat map for best obtained scenario
heatMapMatrix = np.loadtxt(open(path_results+'speedHeatMap'+str(indx_best)+'.csv', 'rt'), delimiter=",")
df_for_smooth = pd.DataFrame(heatMapMatrix[1:,:]*3.6)
df = gaussian_filter(df_for_smooth, sigma=0.5)
ax = sns.heatmap(df, cmap="jet_r", vmin=0, vmax=120, cbar_kws={'label': 'Speed [km/h]'},rasterized=True)
ax.invert_yaxis()
xticks_labels = [0,1,2,3,4,5,6,7,8]#[0, 2000, 4000, 6000, 8000]
plt.xticks(np.arange(0,161,20), xticks_labels, rotation=0)
yticks_labels = [0, .25, .5, .75, 1, 1.25, 1.5]
plt.yticks(np.arange(0,109,18), yticks_labels, rotation=0)
plt.ylabel('Time [h]')
plt.xlabel('Distance [km]')
plt.title('BEST-VSL (spatioltemporal flow speed)')
plt.savefig("Speed_heatMap_DWL_VSLmoving_MD"+str(indx_best)+".png",format='png')
plt.savefig("Speed_heatMap_DWL_VSLmoving_MD"+str(indx_best)+".pdf",format='pdf')
plt.show()
# ================== END Plot Speed heat map for best obtained scenario


# ================== Plot Speed Limit heat map for best obtained scenario
heatMapMatrix = np.loadtxt(open(path_results+'speedLimitHeatMap'+str(indx_best)+'.csv', 'rt'), delimiter=",")
df = pd.DataFrame(heatMapMatrix[1:,:]*3.6)
ax = sns.heatmap(df, cmap="jet_r", vmin=0, vmax=120, cbar_kws={'label': 'Speed limit [km/h]'},rasterized=True)
ax.invert_yaxis()
xticks_labels = [0,1,2,3,4,5,6,7,8] #[0, 2, 4, 6, 8]
plt.xticks(np.arange(0,161,20), xticks_labels, rotation=0)
yticks_labels = [0, .25, .5, .75, 1, 1.25, 1.5]
plt.yticks(np.arange(0,37,6), yticks_labels, rotation=0)
plt.ylabel('Time [h]')
plt.xlabel('Distance [km]')
plt.title('SPEED LIMITS and VSL zones activation')
plt.savefig("SpeedLimit_heatMap_DWL_VSLmoving_MD"+str(indx_best)+".png",format='png')
plt.savefig("SpeedLimit_heatMap_DWL_VSLmoving_MD"+str(indx_best)+".pdf",format='pdf')
plt.show()
# ================== END Plot Speed limit heat map for best obtained scenario