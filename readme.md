# A Free Energy Principle approach to modelling swarm behaviors

*Note: clone single branch to avoid downloading unnecessary asset files (GIFs and Images)*

```Bash
git clone --single-branch --branch master https://github.com/mimocha/fep-swarm.git
```

---

### Contents

1. [**About**](#about)
2. [**GIFs**](#here-are-some-cool-gifs)
3. [**Usage**](#usage)
4. [**Simulation Parameters**](#simulation-parameters)
5. [**Cell Properties**](#cell-properties)

---

## About

```
This repository contains the source code to the MSc dissertation 
"A Free Energy Principle approach to modelling swarm behaviors"
presented for the degree of MSc Intelligent and Adaptive Systems 
at the University of Sussex.

Chawit Leosrisook  
School of Engineering and Informatics  
University of Sussex  
2019-2020
```

This project applies the Free Energy Principle to modelling self-organizing swarms, largely based on the paper [*"On Markov blankets and hierarchical self-organisation"*, Palacios et al. (2020)](https://doi.org/10.1016/j.jtbi.2019.110089)

This work reimplements the model presented by Palacios from scratch, with the full codes provided in this repository. Here are some examples of the simulation in action.

<p align="center">
	<b>First, an example simulation setup:</b>
</p>

<p align="center">
	<img src="https://github.com/mimocha/fep-swarm/blob/assets/img/heatmap.jpg" alt="Sample Simulation Setup" width="600"/>
</p>

The top-left box shows the 2D particle agents, color coded based on their "beliefs". These beliefs are about what type of agent they are; red, green or blue.
The parameters of the simulation is shown on top of this figure.

`N = number of agents | k = "learning rates" | dt = step size | Time = time of simulation`

The other three boxes shows a heatmap of signals being outputted by each type of agent.

Note that the signals roughly corresponds to the location of each agent, but doesn't match perfectly (signals seems mixed). This is because the agents have "mixed beliefs" which changes over time. This is easier to understand when you see them in action below. (Also, see the paper itself.)

### Here are some cool GIFs.

Things get quite interesting with larger number of agents.  
They're quite fun to play around with, so I provide a lot of options.

<p align="center">
<img src="https://github.com/mimocha/fep-swarm/blob/assets/vid/demo-1.gif" alt="Demo-1" width="400" />
<img src="https://github.com/mimocha/fep-swarm/blob/assets/vid/demo-2.gif" alt="Demo-2" width="400" />
<img src="https://github.com/mimocha/fep-swarm/blob/assets/vid/demo-3.gif" alt="Demo-3" width="400" />
<img src="https://github.com/mimocha/fep-swarm/blob/assets/vid/demo-4.gif" alt="Demo-4" width="400" />
</p>

---

&nbsp;

## Usage

Files under the `demo/` directory are standalone files that should run on all versions of MATLAB. These should run on their own, as-is, so treat them like a quick start.

The `main.m` file (under `src/`) is the "main test-bench" for when you want to change things around. *(Nothing stops you from changing the demo files, but those replicates the experiments in my dissertation.)*
The helper functions are split into its own files with some docs, so you can use the MATLAB `help <function>` to get a quick explanation.

I've set the scripts in a way that the **simulation parameters** and initial **cell properties** are in their own sections. Change those as you see fit. There shouldn't be anything else to change further down inside the code, unless you are about to change the model itself.

---

### Simulation Parameters:

Save GIF of the current sim: `GIF = <true/false>`  
Filename (and location) of the GIF: `filename = "..."`

`drawInt = <int>` -- Figure drawing interval. **Units in simulation ticks**, with lower number means higher frequency (1 is draw every tick). Simulation can get quite slow with lower number, so setting to `10 / 20 / 50` is nice for most cases. `100` and higher is quite crude, but good if you want to run longer / larger sims.

`axRange = <float>` -- Simulation viewport range. Self-explanatory. Can slow down the simulation in combination with the heatmaps (they are quite slow).

`boundary = <true/false>` -- Simulation boundary condition. Locks agent to within viewport.

`axLock = <true/false>` -- "Axis Lock". Tries to move agents to the center of the viewport. This option is nice when you don't want the `boundary` option, but still want to keep track of the agents some how. This option **tracks the position of blue cells**, and moves all agents relative to that. If there are no blue cells, it just stops tracking.

`hmSpace = <float>` -- Heatmap grid spacing. Lower number gives finer heatmap grid (and smoother look). But gets very computationally expensive real fast. I recommend `0.2` for most usage, as it's fast enough to be viewed smoothly in real time (depends on PC). Use `0.05` If you want to take nice screenshots / GIFs (like the above demos), but I don't recommend any smaller. Also, I recommend increasing `drawInt` / decreasing `axRange` to help with simulation speed, if you really want fine-grained heatmaps.

`gqSpace = <float>` -- Gradient Quiver spacing. A variation of the `hmSpace` option, found in `gradient_quiver.m` (demo). Sets the spacing for "gradient vectors". Not very useful demo, but fun to look at nonetheless.

<p align="center">
<img src="https://github.com/mimocha/fep-swarm/blob/assets/vid/demo-gq.gif" alt="Gradient Quiver Example" width="400" />
</p>

`showMoves = <true/false>` -- Show movement arrows on agents. Set to true to enable movement vectors to be drawn on agents. Arrow size is relatively scaled, so you can get jerky movement arrows when the agents are very still.

`Number of cells` -- Self-explanatory. Set the initial agent types as needed. However this doesn't matter if you plan to use randomized `mu` later on.

`dt = <float>` -- Simulation step size. Self-explanatory. I typically use `0.01` for fine simulation, or `0.05` for coarser sims.

`tLimit` -- Simulation time limit. **Units in seconds**.

---

### Cell Properties:

`k_a` -- Action "learning rate". Decrease to slow down agent movement.  
`k_mu` -- Inference "learning rate". Decrese to slow down agent inference (color change rate).

`p_x` and `p_y` -- Generative Parameters. Really need to read the paper to understand this. Effectively encodes how different type of agents interact with one another.

`mu` -- Initial internal states. [3,N] matrix. One row per cell type (3 types total). Internal agent values about what they believe. Changes to beliefs by taking a softmax of this value.

`sigma_mu` -- Initial agent belief. Softmax of `mu`. This is used to color code agents.

`psi_x` -- Agent positions. [2,N] matrix, top row is x-coordinate, bottom row is y-coordinate.

`psi_y` -- Agent signals. [3,N] matrix. One row per signal type (3 types total).

*Changing the following does nothing (gets overwritten anyway):*

`s_x` and `s_y` -- Agent sensory data.

`a_x` and `a_y` -- Agent actions.

`epsilon_x` and `epsilon_y` -- Agent prediction error.

---

<p align="center"> 
	<a href="#top"> 
		<code>Back to top</code>
	</a>
</p>