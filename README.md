# SMGP_BCI_EEG

## Overview
This repository contains the simulation code for my thesis project titled:

**"Bayesian Classification with Split-and-Merge Gaussian Process (SMGP) Prior in EEG-based Brain-Computer Interfaces"**

The project focuses on modeling EEG signals using the Split-and-Merge Gaussian Process (SMGP) prior and compares its classification performance against reference methods, including Bayesian Linear Discriminant Analysis (BLDA) and Stepwise Linear Discriminant Analysis (swLDA).

> **Note:** Due to data privacy restrictions, real participant EEG data cannot be shared. This repository includes only the simulation framework for data generation, model fitting, and performance evaluation.

---

## Project Structure & Workflow

### 1. Global Parameter Setup
- **`SIM_generate.py`**  
  Define global parameters for simulations (e.g., number of channels, trials, noise levels).

---

### 2. Single-Channel Simulation Workflow
- **Model Sampling:**
  - `SIM_single_pyro_GP.py` — Run SMGP-based Bayesian classification.
  - `SIM_single_reference.py` — Run BLDA as a reference method.
- **swLDA Analysis:**
  - `SIM_single_swLDA_fit.m` — Fit swLDA model (MATLAB).
  - `SIM_single_swLDA_predict.py` — Predict using swLDA.
- **Visualization:**
  - `SIM_single_visual_hard.py` — Visualize SMGP and BLDA results.

---

### 3. Multi-Channel Simulation Workflow
- **Model Sampling:**
  - `SIM_multi_pyro_GP.py` — Run SMGP-based Bayesian classification.
  - `SIM_multi_reference.py` — Run BLDA as a reference method.
- **swLDA Analysis:**
  - `SIM_swLDA_fit.m` — Fit swLDA model (MATLAB).
  - `SIM_multi_swLDA_predict.py` — Predict using swLDA.
- **Visualization:**
  - `SIM_multi_visual_hard.py` — Visualize SMGP and BLDA results.

---

### 4. Supporting Functions
- **Python Utilities:**
  - `generate_Func.py`, `model_Func.py`, `multi_visual_Func.py`, `single_visual_Func.py`, `source.py`  
    Core functions for data generation, model operations, and visualization.
- **MATLAB Utilities:**
  - `prepareDataForSWLDA.m`, `prepareSingleDataForSWLDA.m`, `trainSWLDAMatlab.m`  
    Data preparation and training scripts for swLDA.

---

## Environment & Dependencies

### Python
- **Version:** 3.11
- **Key Packages:**
  - `numpyro`
  - `numpy`
  - `matplotlib` (for visualization)
  - Other standard libraries

> Install dependencies using:
```bash
pip install numpyro numpy matplotlib
