# Distributed W-Learning based Spatio-Temporal Variable Speed Limit (DWL-ST-VSL)
This git repository is linked to the research papers entitled "_Dynamic Variable Speed Limit Zones Allocation Using
Distributed Multi-Agent Reinforcement Learning_", and "_Extended Variable Speed Limit control using Multi-agent
Reinforcement Learning_"<br/>

This repository includes all the files required to run the simulations used to produce the results (medium traffic load) presented in the papers:

## (SCENARIO-I) Simulation of dynamic variable speed limit zones allocation using two collaborative agents controlling two segments upstream of the congestion area in SUMO:
Notes: Medium traffic load  
* SUMO synthetic motorway model and traffic scenario,
* python code for:<br/>

(I) Machine learnin code for training of Distributed W-Learnin based Multi-Agent Reinforcement Learning (RL)-based VSL controller (DWL-ST-VSL),<br/>
(II) Code (Python-TraCI) for adaptive VSL zones allocation in microscopic simulator SUMO,<br/>
(III) Analytics (writting results **traffic parameters**, **controller parameters**, **plotting spatiotemporal traffic characteristics and dynamic VSL zones allocation** (comparison between NO-VSL and DWL-ST-VSL).

Please use the following citation when referencing our work. Thanks!
>@INPROCEEDINGS{KusicITSC2021,
> author={Kušić, Krešimir and Ivanjko, Edouard and Vrbanić, Filip and Gregurić, Martin and Dusparic, Ivana},
> booktitle={2021 IEEE International Intelligent Transportation Systems Conference (ITSC)}, 
> title={Dynamic Variable Speed Limit Zones Allocation Using Distributed Multi-Agent Reinforcement Learning}, 
> year={2021},
> volume={},
> number={},
> pages={3238-3245},
> doi={10.1109/ITSC48978.2021.9564739}
>}

>@INPROCEEDINGS{KusicITSC2020,
> author={K. {Kušić} and I. {Dusparic} and M. {Guériau} and M. {Gregurić} and E. {Ivanjko}},
> booktitle={2020 IEEE 23rd International Conference on Intelligent Transportation Systems (ITSC)}, 
> title={Extended Variable Speed Limit control using Multi-agent Reinforcement Learning}, 
> year={2020},
> volume={},
> number={},
> pages={1-8},
> doi={10.1109/ITSC45102.2020.9294639}
>}

## (SCENARIO-II) four agents:
Notes: Please contact us (see details in the article below)!

>@Article{math9233081,
> AUTHOR = {Kušić, Krešimir and Ivanjko, Edouard and Vrbanić, Filip and Gregurić, Martin and Dusparic, Ivana},
> TITLE = {Spatial-Temporal Traffic Flow Control on Motorways Using Distributed Multi-Agent Reinforcement Learning},
> JOURNAL = {Mathematics},
> VOLUME = {9},
> YEAR = {2021},
> NUMBER = {23},
> ARTICLE-NUMBER = {3081},
> ISSN = {2227-7390},
> DOI = {10.3390/math9233081}
>}



# How To setup entire learning-based simulation framework (SUMO-Python-Distributed W-Learning (DWL) algorithm
Note: we need SUMO and Python (with some additional packages), If you follow carefully steps everything should run smoothly **(Update the necessary paths!)**
     

## Python setup:
(I) Install Python (we tested DWL-ST-VSL in version 3.10.4)<br/>
(II) Open cmd and navigate to the directory where you want to create Python virtual environment (can be a path to our folder you downloaded) and run commands 
below in cmd to create Python virtual environment using terminal command (Note: if you have multiple pythons installed specify a version):<br/>

create virtual environment
```
python -m venv venvDWLSTVSL
```
activate virtual environment
```
venvDWLSTVSL\Scripts\activate
```
update pip
```
python -m pip install --upgrade pip
```
install necessary libraries from **requirements.txt** file
```
pip install -r requirements.txt 
```
## SUMO setup:
Note: in the paper, we used an older version, so the results might slightly vary depending on the SUMO version as well as on the process of training the machine-learning agents<br/>
Instal SUMO (we tested DWL-ST-VSL in version Latest Development Version SUMO1.13.0 for Windows 64-bit (May 30 2022 23:15:46 UTC)

# Run Scenario I

activate virtual environemt
```
C:\Users\kkusic\Desktop\DWL-ST-VSL>venvDWLSTVSL\Scripts\activate
```
run simulation script:
```
(venvDWLSTVSL) C:\Users\kkusic\Desktop\DWL-ST-VSL>python DWL_ST_VSL_2Agents.py
```
when simulations are finished, plot results and print analytics -> run the script:
```
(venvDWLSTVSL) C:\Users\kkusic\Desktop\DWL-ST-VSL>python plot_results.py
```


## Users
*
If you use DT-GM or its tools to create a new one, we would be glad to add you to the list.
You can send an email with your name and affiliation to kusic.kresimir@hes-so.ch or kresimir.kusic@unizg.fpz.hr

As well, feel free to contact us if you experience any problems, so we can improve or add additional comments to make DWL-ST-VSL easier to use.
