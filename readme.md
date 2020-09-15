# A Free Energy Principle approach to modelling swarm behaviors

```
This repository contains the source code to the MSc dissertation 
"A Free Energy Principle approach to modelling swarm behaviors"
presented for the degree of MSc Intelligent and Adaptive Systems 
#at the University of Sussex.

Chawit Leosrisook  
School of Engineering and Informatics  
University of Sussex  
2019-2020
```

---

## About

This project applies the Free Energy Principle to modelling self-organizing swarms, largely based on the paper [*"On Markov blankets and hierarchical self-organisation"*, Palacios et al. (2020)](https://doi.org/10.1016/j.jtbi.2019.110089)

This work reimplements the model presented by Palacios from scratch, with the full codes provided in this repository. Here are some examples of the simulation in action.

<p align="center">
	<b>First, an example simulation setup:</b>
</p>

<p align="center">
	<img src="https://github.com/mimocha/fep-swarm/blob/assets/img/heatmap.png" alt="Sample Simulation Setup" width="600"/>
</p>

The top-left box shows the 2D particle agents, color coded based on their "beliefs". These beliefs are about what type of agent they are; red, green or blue.
The parameters of the simulation is shown on top of this figure.

`N = number of agents | k = "learning rates" | dt = step size | Time = time of simulation`

The other three boxes shows a heatmap of signals being outputted by each type of agent.

Note that the signals roughly corresponds to the location of each agent, but doesn't match perfectly (signals seems mixed). This is because the agents have "mixed beliefs" which changes over time. This is easier to understand when you see them in action below. (Also, see the paper itself.)

---

**Here are some cool GIFs.**

First, a small simulation.

<p align="center">
<img src="https://github.com/mimocha/fep-swarm/blob/assets/vid/demo-1.gif" alt="Demo-1" width="600" />
</p>

Things gets a lot more interesting with larger number of agents

<p align="center">
<img src="https://github.com/mimocha/fep-swarm/blob/assets/vid/demo-2.gif" alt="Demo-2" width="600" />
</p>

They're quite fun to play around with, so I provide a lot of options.

<p align="center">
<img src="https://github.com/mimocha/fep-swarm/blob/assets/vid/demo-3.gif" alt="Demo-3" width="400" />
<img src="https://github.com/mimocha/fep-swarm/blob/assets/vid/demo-4.gif" alt="Demo-4" width="400" />
</p>

---

## Usage

Files under the `demo` directories are standalone files that should run on all versions of MATLAB. These can run on their own, as-is, so treat it like a quick start.

The `main.m` file is the "main test-bench" for when you want to change things around. (Nothing stops you from changing the demo files, but those replicates the experiments in my dissertation.)

The helper functions are split into its own files, so you can use the MATLAB `help <function>` to get a quick explanation for each file.

I've set the scripts in a way that the **simulation parameters** and initial **cell properties** are in their own sections. Change those as you see fit. There shouldn't be anything else to change further down inside the code, unless you are about to change the model itself.