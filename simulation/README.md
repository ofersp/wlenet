# Producing Simulations of Lensed Galaxy Stamps

Here `galsim_simulation.py` can be used in one of the following ways to generate simulations:

1. By instantiating a wlenet.simulation.galsim_simulation.GalsimSimulation object (argument/json based configuration).
2. By using `galsim_map_reduce.py` to generate a set of command lines to be executed either locally or on a cluster (json based configuration).
3. By directly running `galsim_simulation.py` in the shell in single-process mode (json/command-line based configuration).
4. By directly running `galsim_simulation.py` in the shell in multi-process mode (json/command-line based configuration). (TODO: implement)
