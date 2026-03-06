# uns_diec_ml

**Machine Learning (PhD in Engineering) — Department of Electronic Engineering and Computing (DIEC)**

This repository contains solutions and implementations for the **Machine Learning** course in the PhD program in Engineering at **DIEC**. It includes **code, experiments, and reports** covering core algorithms and advanced methods for research and practice.

---

## Contents

- **Practical Works (TPs)** implemented in **Python**
- **Jupyter Notebooks** with experiments, plots, and analysis
- Coverage of **core + advanced ML** methods (supervised/unsupervised, model selection, evaluation, etc.)
- Reusable utilities for **preprocessing, metrics, and visualization**
- Reproducible setup via **Conda**

---

## Experimental Environment

- **CPU:** AMD Ryzen 7  
- **RAM:** 64 GB  
- **OS:** Windows 10 Pro (64-bit)  
- **Python:** 3.12 (Anaconda 3)  
- **Workbench:** Jupyter Notebook / JupyterLab  

---

## Conda setup (Windows + Anaconda)

```bash
conda env create -f environment.yml
conda activate am_uns
```

Register the kernel for Jupyter:

```bash
python -m ipykernel install --user --name am_uns --display-name "Python (am_uns)"
```

Launch notebooks:

```bash
jupyter notebook
```
---

## License

Academic / educational use. If you reuse parts of this repository, please provide attribution.

---

## Author

**Pablo Nicolás Ramos**  
PhD Program in Engineering — Machine Learning (DIEC)