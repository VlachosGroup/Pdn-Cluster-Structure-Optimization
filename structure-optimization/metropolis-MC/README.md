# Metropolis MC 

## Usage

Metropolis is performed by executing `run_metropolis.py`

### Simulation parameters 
- rseed, random seed
- T, temperature
- n_seeds, the number of Pd atoms
- iterations, the number of MC iterations 

### Output files describing a Metropolis MC trajectory
- `metropolis_outputs/metropolis_Tk_rseed.p`, the pickle file for accepted configurations, accpetance ratios and energies

### Post-processing 
- Use `read_metropolis` to generate trajectory plots and visualize atomic configurations
