# Public accompanying code to [``Sequence-Space Jacobians of Life Cycle Models''](https://www.bencebardoczy.com/files/lc-ssj.pdf)

## By [Bence Bardoczy](https://www.bencebardoczy.com/), [Akshay Shanker](https://akshayshanker.net/) and [Mateo Velasquez-Giraldo](https://mateovg.com/)

**Launch an interactive demo:**     [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Mv77/LC-SSJ_public/HEAD?urlpath=%2Fdoc%2Ftree%2FCode%2Fdemo.ipynb)

This repository contains code that implements and demonstrates our method for finding Sequence-Space Jacobians of life cycle models. The main file is [`Code/demo.ipynb`](https://github.com/Mv77/LC-SSJ_public/blob/main/Code/demo.ipynb). It is a complement to the "Cookbook" appendix of the paper and has code to reproduce all its results and figures. The code further demonstrates how to integrate the Jacobians that we calculate with the [sequence-jacobian](https://github.com/shade-econ/sequence-jacobian) toolkit and perform equilibrium analyses. It uses a simple pension reform as an illustrative example.

We will try to keep the code updated and provide better comments as best we can. Contact us if you have issues.

Please cite the paper if you find this code useful. Here is a suggested `.bib` entry **(to be updated soon)**:
```
@online{bardoczy_sequence-space_2025,
  type = {SSRN Scholarly Paper},
  title = {Sequence-{{Space Jacobians}} of {{Life Cycle Models}}},
  author = {Bardóczy, Bence and Velásquez-Giraldo, Mateo},
  date = {2025-05-29},
  number = {5274675},
  eprint = {5274675},
  eprinttype = {Social Science Research Network},
  location = {Rochester, NY},
  doi = {10.2139/ssrn.5274675},
  url = {https://papers.ssrn.com/abstract=5274675},
  urldate = {2025-06-02},
  langid = {english},
  pubstate = {prepublished},
}

```

Our code uses and builds on the [sequence-jacobian](https://github.com/shade-econ/sequence-jacobian/tree/master) and [HARK](https://github.com/econ-ark/HARK) toolkits.

All the required packages to run the code in this repository can be installed using the provided environment file, `environment.yml`, by running
```
conda env create -f environment.yml
```
