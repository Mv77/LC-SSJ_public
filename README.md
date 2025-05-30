# Public acompanying code to [``Sequence-Space Jacobians of Life Cycle Models''](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5274675)

## By [Bence Bardoczy](https://www.bencebardoczy.com/) and [Mateo Velasquez-Giraldo](https://mateovg.com/)

**Launch an interactive demo:**     [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Mv77/LC-SSJ_public/HEAD?urlpath=%2Fdoc%2Ftree%2Fdemo.ipynb)

This repo contains an code that implements and demonstrates our method for finding Sequence-Space Jacobians of life cycle models.

It has code to [reproduce all the figures](https://github.com/Mv77/LC-SSJ_public/blob/main/Paper_figures.py) and [experiments](https://github.com/Mv77/LC-SSJ_public/blob/main/LC_IH_benchmarks.py) reported in the paper, as well as a [standalone demonstration](https://github.com/Mv77/LC-SSJ_public/blob/main/demo.ipynb) in a jupyter notebook.

Please cite the paper if you find this code useful. Here is a suggested `.bib` entry:
```
@online{bardoczy_sequence-space_2025,
  type = {SSRN Scholarly Paper},
  title = {Sequence-{{Space Jacobians}} of {{Life Cycle Models}}},
  author = {Bardoczy, Bence and Velasquez-Giraldo, Mateo},
  date = {2025-05-29},
  number = {5274675},
  eprint = {5274675},
  eprinttype = {Social Science Research Network},
  location = {Rochester, NY},
  url = {https://papers.ssrn.com/abstract=5274675},
  urldate = {2025-05-29},
  langid = {english},
  pubstate = {prepublished},
  keywords = {General Equilibrium,Heterogeneous Agent Models,Life Cycle Dynamics,Sequence Space Jacobians},
}
```

Our code uses and builds on the [sequence-jacobian](https://github.com/shade-econ/sequence-jacobian/tree/master) and [HARK](https://github.com/econ-ark/HARK) toolkits.

All the required packages to run the code in this repository can be installed using the provided environment file, `environment.yml`, by running
```
conda env create -f environment.yml
```