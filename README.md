# Neural Co-state Regulator Implementation on Bicycle Model

This repository contains the bicycle-model implementation of a Neural Co-state Regulator (NCR) workflow. The code trains a Co-state Neural Network (CoNN) to approximate optimal co-state trajectories, then uses the learned co-states to compute real-time control inputs for a bicycle model with input constraints.

This work is related to the paper **"Neural Co-state Regulator: A Data-Driven Paradigm for Real-time Optimal Control with Input Constraints"**, published at CDC 2025. More information about the original project can be found on the [project website](https://lihanlian.github.io/neural_co-state_regulator/).

## Structure

- `config_template.py`  
  Template configuration file.

- `config.py`  
  Local experiment configuration. This file controls sampling time, prediction horizon, training epochs, cost weights, input constraints, initial states, and simulation settings.

- `bi_train.py`  
  Trains the Co-state Neural Network for the bicycle model.

- `bi_sim_ncr.py`  
  Runs NCR simulation using a trained checkpoint.

- `bi_mpc.py`  
  Defines and runs the bicycle Model Predictive Control baseline.

- `bi_mpc_rnd_traj_sim.py`  
  Runs MPC simulations over sampled/random initial conditions.

- `bi_ncr_rnd_traj_sim.py`  
  Runs NCR simulations over sampled/random initial conditions.

- `bi_utils.py`  
  Shared utility functions for bicycle dynamics, RK4 integration, QP solving, CBF constraints, plotting, and animation.

- `run.py`  
  Windows helper script that runs both training and simulation.

- `run_cont.py`  
  Windows helper script that continues training from a previous checkpoint, then runs NCR simulation.

## Results

**NCR Case B: Seen initial conditions and zero reference**
<p align="center">
  <img alt="Image 1" src="bi_figs\bi_ncr_N100_b.png" width="45%" />
  <img alt="Image 2" src="bi_figs\bi_animation_b.gif" width="45%" />
</p>

** NCR sampled/random initial conditions simulation **
<p align="center">
  <img alt="Image 1" src="bi_figs\rate0.62_N100_seed1_thre0.4_Tsim20_2-6-2-6.png" width="80%" />
</p>

** MPC sampled/random initial conditions simulation **
<p align="center">
  <img alt="Image 1" src="bi_figs\MPC_rate1.00_N100_dt0.05_Tsim20_dtmpc0.05_dtpred0.05_seed1_thre0.4_0-6-0-6.png" width="80%" />
</p>

## Setup

Clone the project and go to project directory
```bash
  python3 -m venv env && source env/bin/activate 
```
```bash
  pip install -r requirements.txt
```
## License

[MIT](./LICENSE)
