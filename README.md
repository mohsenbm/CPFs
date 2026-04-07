# Corrected Product Formulas

[![arXiv](https://img.shields.io/badge/arXiv:2409.08265v3-a62c2b.svg)](https://arxiv.org/abs/2409.08265)
[![DOI](https://img.shields.io/badge/DOI:10.48550/arXiv.2409.08265-0077C8.svg)](https://arxiv.org/abs/2409.08265)

This repository contains all the codes used for [arXiv:2409.08265](https://arxiv.org/abs/2409.08265). It contains implementations of Product Formulas (PFs) and Corrected Product Formulas (CPFs), and code to perform time evolution. We use various Hamiltonians for classcial simulations. Fo quantum hardware implementations, we use the Ising model and benchmark its time evolution obtained via CPFs and PFs against the exact time evolution of the Ising model.

### Installation

To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

Add a `.env` file with your IBM credentials:

```bash
IBM_TOKEN="<your-token>"
IBMQ_CHANNEL="<your-channel>"
IBMQ_INSTANCE="<your-instance>"
IBMQ_BACKEND="<your-backend>"
```

In our case, the IBMQ channel is `ibm_quantum`, the IBMQ instance is `pinq-quebec-hub/univ-toronto/matterlab`, and the backend is `ibm_quebec`.

### Usage

To reproduce the results of the paper, run the code in the `notebooks` folder. The main logic of the code is in the `src` folder. The date generated from hardware experiments and numerical simulations are in `data` the folder.

### Contributors

Numerical/classcial simulations: Mohsen Bagherimehrab

Quantum hardware implementations: Mohsen Bagherimehrab and Luis Mantilla Calderon
(We acknowledge useful discussions with Mohammad Ghazi Vakili on quantum hardware implementations)

### Citation:
If you use this implementation or results from the paper, please cite our work as
```bash
@misc{BMB+25,
      title={Faster Algorithmic Quantum and Classical Simulations by Corrected Product Formulas}, 
      author={Mohsen Bagherimehrab and Luis Mantilla Calderon and Dominic W. Berry and Philipp Schleich and Mohammad Ghazi Vakili and Abdulrahman Aldossary          and Jorge A. Campos Gonzalez Angulo and Christoph Gorgulla and Alan Aspuru-Guzik},
      year={2025},
      eprint={2409.08265},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2409.08265},
      doi={10.48550/arXiv.2409.08265}
}
```