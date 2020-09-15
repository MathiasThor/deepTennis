# Deep Reach

### Introduction

This project is about training two agents to control rackets to bounce a ball over a net. The project uses the Unity ML-Agents Tennis Environment. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it gets a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

GIF HERE!

The agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,
- After each episode, we add up the rewards that each agent received (without discounting) to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these two scores.
- This yields a single score for each episode.

The task is considered solved when the average (over 100 episodes) of those scores is at least +0.5.

### Getting started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

2. Place the file in the unity_simulation/ folder, and unzip (or decompress) the file.
3. Follow the dependencies guide at the end of this readme (if you haven't already).
4. Navigate to the project root folder: ```bash source activate drlnd && jupyter notebook ```
5. Specify the patch to the environment in the first cell of `deepTennis.ipynb`.
6. Run all the cells to and with *train PPO agent* in `deepTennis.ipynb` to start learning.
7. Run the *Plot score* to plot the average score against episodes from the learning session.
8. Run the last cell (*load and test trained agent*) to show the learned agent.


### Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
	
3. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/MathiasThor/BananaCollector.git
cd BananaCollector/python
pip install .
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment using the drop-down `Kernel` menu. 

<img src="https://sdk.bitmoji.com/render/panel/e68fcf49-ccb9-4878-a8b3-21834fdcef55-fdafebb8-6b14-4983-9cbb-a539f77ab069-v1.png?transparent=1&palette=1" width="250" height="250">



