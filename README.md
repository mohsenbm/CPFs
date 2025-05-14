# Corrected Product Formulas

### Description

This repository contains all the codes used for [arXiv:2409.08265](https://arxiv.org/abs/2409.08265) [1]. It contains multiple implementations of Corrected Product Formulas (CPFs) and code to perform time evolution of the Isign model. We benchmark the time evolution obtained via CPFs and PFs against the exact time evolution of the Ising model obtained following the method described in Refs. [2,3].

### Installation

To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

Add a `.env` file with your IBM token:

```bash
IBM_TOKEN="<your-token>"
```

### References

[1] Bagherimehrab, Mohsen, Luis Mantilla Calderon, Dominic W. Berry, Philipp Schleich, Mohammad Ghazi Vakili, Abdulrahman Aldossary, Jorge A. Angulo, Christoph Gorgulla, and Alan Aspuru-Guzik. "Faster algorithmic quantum and classical simulations by corrected product formulas." arXiv preprint arXiv:2409.08265 (2024).

[2] F. Verstraete, J. I. Cirac, and J. I. Latorre, “Quantum circuits for strongly correlated quantum systems,” Phys. Rev. A 79, 032316 (2009).

[3] A. Cervera-Lierta, “Exact Ising model simulation on a quantum computer,” Quantum 2, 114 (2018).
