# Corrected Product Formulas

[![arXiv](https://img.shields.io/badge/arXiv:2409.08265v3-a62c2b.svg)](https://arxiv.org/abs/2409.08265)
[![DOI](https://img.shields.io/badge/DOI:10.48550/arXiv.2409.08265-0077C8.svg)](https://arxiv.org/abs/2409.08265)

This repository contains all the codes used for [arXiv:2409.08265](https://arxiv.org/abs/2409.08265) [1]. It contains multiple implementations of Corrected Product Formulas (CPFs) and code to perform time evolution. We use various Hamiltonians for classcial simulations. Fo quantum hardware implementations, we use Ising model and benchmark its time evolution obtained via CPFs and PFs against the exact time evolution of the Ising model obtained following the method described in Refs. [2,3].

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

To reproduce the results of the paper, run the code in the `notebooks` folder. The main logic of the code is in the `src` folder. 

If you want to run the code using a docker container, you can first create the docker image with

```bash
docker build -t cpfs-image .
```

Then, run the container with

```bash
docker run --rm --env-file .env cpfs-image
```

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

### References

[1] Bagherimehrab, Mohsen, Luis Mantilla Calderon, Dominic W. Berry, Philipp Schleich, Mohammad Ghazi Vakili, Abdulrahman Aldossary, Jorge A. Angulo, Christoph Gorgulla, and Alan Aspuru-Guzik. "Faster algorithmic quantum and classical simulations by corrected product formulas." arXiv preprint arXiv:2409.08265 (2024).

[2] F. Verstraete, J. I. Cirac, and J. I. Latorre, “Quantum circuits for strongly correlated quantum systems,” Phys. Rev. A 79, 032316 (2009).

[3] A. Cervera-Lierta, “Exact Ising model simulation on a quantum computer,” Quantum 2, 114 (2018).
