# crystal-x

## Project description

crystal-x provides a steady-state simulation for the Czochralski method using the finite element library [dolfinx](https://github.com/FEniCS/dolfinx). In the simulation, a time-harmonic induction equation is coupled with the heat equation. Additionally, the crystalization interface is adjusted to fit the geometry.

The project is developed and maintained by the [**Model experiments group**](https://www.ikz-berlin.de/en/research/materials-science/section-fundamental-description#c486) at the Leibniz Institute for Crystal Growth (IKZ).

### Referencing

If you use this code in your research, please cite our article:

> TODO
## Prerequisites
crystal-x requires Python >= 3.10. and dolfinx version 0.5.0. Additionally, numpy, scipy, pyyaml, meshio and objectgmsh are required.

The easiest way to run this code is to use the dockerfile provided in the repository.

## Usage
The code is split into different submodules. The geometry is created in the geometry files. The weak forms are created in the [equations folder](https://github.com/nemocrys/crystal-x/tree/main/crystalx/steadystate/equations). Additionally, helper functions and the calculation of the interface are found in the [auxiliary methods file](https://github.com/nemocrys/crystal-x/blob/main/crystalx/steadystate/auxiliary_methods.py).

## Example
A steady-state example is provided [here](https://github.com/nemocrys/crystal-x/blob/main/examples/steady_state_simulation.py), where the parameters are set [yml file](https://github.com/nemocrys/crystal-x/blob/main/examples/setup_steady_state_simulation.yml).

## License

crystal-x is published under the [GPLv3 license](https://www.gnu.org/licenses/gpl-3.0.html).

## Acknowledgements

[This project](https://www.researchgate.net/project/NEMOCRYS-Next-Generation-Multiphysical-Models-for-Crystal-Growth-Processes) has received funding from the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation programme (grant agreement No 851768).

<img src="https://raw.githubusercontent.com/nemocrys/nemoblock/master/EU-ERC.png">

## Contribution

Any help to improve this package is very welcome!
