import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# matplotlib.rcParams['pdf.fonttype'] = 3 # default
# matplotlib.rcParams['ps.fonttype'] = 3 # default
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sim_start=0
sim_end=14000
win_size=10

path_to_DWL2='C:/Users/kusic/Desktop/DWL-ST-VSL/Results/'
path_to_MD_NO_VSL='C:/Users/kusic/Desktop/DWL-ST-VSL/Results/'

df_DWL2=pd.DataFrame(columns=['TTS'])

for i in range(sim_start,sim_end+1,20):

    data = pd.read_excel(path_to_DWL2+"controlFile"+str(i)+".xlsx")
    df_DWL2 = pd.DataFrame(np.vstack((df_DWL2, data.loc[data.index[-1], "TTSsim"])))


df=pd.DataFrame({'DWL2': df_DWL2.iloc[:,0]})
df['DWL2'] = df.iloc[:,0].rolling(window=win_size).mean()

TTS_DWL2 = np.array([])
TTS_DWL2 = np.append(TTS_DWL2,df['DWL2'][(win_size-1):])

df_NO_VSL=pd.DataFrame(columns=['TTS'])
data = pd.read_excel(path_to_MD_NO_VSL+"NO_VSL_controlFile0.xlsx")
df_NO_VSL = pd.DataFrame(np.vstack((df_NO_VSL, data.loc[data.index[-1], "TTSsim"])))

t_TTS=np.arange(sim_start, (TTS_DWL2.size), 1)

plt.hlines(np.mean(df_NO_VSL.iloc[0,0]),0,sim_end+1,linestyle='--',color='r',label='NO-VSL',linewidth=2)
plt.plot(t_TTS, TTS_DWL2[:],linestyle='-',color='dodgerblue',label='DWL2-ST-VSL',linewidth=1.0)


plt.legend(loc='best')
plt.xlabel('Number of simulations')
plt.xlim(0, 7*98.857)
# x=[0, 100, 200, 300, 400, 500, 600, 700]
x = [0, 98.857, 2*98.857, 3*98.857, 4*98.857, 5*98.857, 6*98.857, 7*98.857] # TTS_DWL2.shape (692,)
x1 = [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000]
plt.xticks(x, x1)
#         plt.figure(figsize=(5,5))
#         plt.ylim(380,420)
plt.ylim(440,470)
plt.ylabel('TTS [veh*h]')
plt.title('TTS - overall network')
plt.grid(axis='y')
# plt.grid(True)

handles, labels = plt.gca().get_legend_handles_labels()
# [0,1,2,3,4]
order = [0,1]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])

# save figure (Total time spent (TTS) == Total travel time)
plt.savefig("TTS_network_MD_DWL2_"+str(sim_end)+".png",format='png')
plt.savefig("TTS_network_MD_DWL2_"+str(sim_end)+".pdf",format='pdf')
plt.clf() # clears the entire current figure with all its axes
#==========================

# plt.show()