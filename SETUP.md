# MLDS Group 5 — Project Setup Guide

This guide explains how to get the project running on your machine from scratch. Follow the steps in order and you should have everything working in under 10 minutes.

---

## What's in This Project

We built five machine learning pipelines across two datasets (Titanic survival and Adult Census income). Each algorithm lives in its own folder:

| Folder | Algorithm | Datasets |
|---|---|---|
| `Decision_Tree/` | Decision Tree | Titanic, Adult Census |
| `SVM/` | Support Vector Machine | Titanic, Adult Census |
| `kNN/` | k-Nearest Neighbours | Titanic, Adult Census, Iris |
| `ILP/` | Inductive Logic Programming (Aleph) | Titanic, Adult Census |
| `Reinforcement_Learning/` | Q-Learning | Titanic |

The raw and cleaned datasets are all in `datasets/`.

---

## Prerequisites

- **Python 3.12** (this is what the lab machines use — other 3.x versions should also work)
- **Jupyter Notebook or JupyterLab**
- **SWI-Prolog** — only needed for the ILP notebooks (see note below)

---

## 1. Clone the Repository

```bash
git clone https://gitlab.surrey.ac.uk/Saad23/mlds-group5-project.git
cd mlds-group5-project
```

---

## 2. Install Python Dependencies

All Python notebooks share the same set of libraries. Install them with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

If you are on a Surrey lab machine these are likely already installed. You can double-check by running:

```bash
python -c "import sklearn, pandas, numpy, seaborn; print('All good')"
```

---

## 3. Running the Notebooks

Open Jupyter from the project root:

```bash
jupyter notebook
```

Then navigate to the relevant folder and open the notebook. **Run all cells from top to bottom** — each notebook is self-contained and will find the datasets automatically regardless of where Jupyter was launched from.

### Decision Tree
```
Decision_Tree/decision_tree_titanic.ipynb   ← Titanic survival
Decision_Tree/decision_tree_adult.ipynb     ← Adult Census income
```
Outputs save to `Decision_Tree/outputs_titanic/` and `Decision_Tree/outputs_adult/`.

### SVM
```
SVM/SVM_Titanic.ipynb
SVM/SVM_Adult_Census.ipynb
```

### kNN
```
kNN/knn_three_datasets.ipynb    ← covers Titanic, Adult Census, and Iris
```

### Reinforcement Learning
```
Reinforcement_Learning/Q-Learning.ipynb
```
Alternatively, run the plain Python script:
```bash
python Reinforcement_Learning/Janani_Q_Learning.py
```

### ILP (Inductive Logic Programming)
```
ILP/ILP_Logic_Based_Learning_Project_final.ipynb
```
> **Note:** The ILP notebook calls SWI-Prolog with the Aleph system. You need SWI-Prolog installed before running it:
> - **Linux/Mac:** `sudo apt install swi-prolog` or `brew install swi-prolog`
> - **Windows:** download the installer from [swi-prolog.org](https://www.swi-prolog.org/Download.html)
>
> The `aleph.pl` file inside `ILP/` is already included — you do not need to download it separately.

---

## 4. Datasets

All datasets are pre-cleaned and ready to use — you do not need to re-run the preprocessing notebooks unless you want to see how the data was prepared.

| File | Description |
|---|---|
| `datasets/raw/train.csv` | Titanic raw training data (Kaggle) |
| `datasets/raw/adult.csv` | Adult Census raw data (UCI) |
| `datasets/cleaned/titanic_X_train.csv` | Titanic features — training set |
| `datasets/cleaned/titanic_X_val.csv` | Titanic features — validation set |
| `datasets/cleaned/titanic_y_train.csv` | Titanic labels — training set |
| `datasets/cleaned/titanic_y_val.csv` | Titanic labels — validation set |
| `datasets/cleaned/X_train.csv` | Adult Census features — training set |
| `datasets/cleaned/X_test.csv` | Adult Census features — test set |
| `datasets/cleaned/y_train.csv` | Adult Census labels — training set |
| `datasets/cleaned/y_test.csv` | Adult Census labels — test set |

If you want to see how the data was preprocessed, the notebooks for that are in `datasets/preprocessing/`.

---

## 5. Expected Outputs

Each notebook saves its plots and result tables automatically when you run it. If a folder does not exist yet it will be created on first run — you do not need to create anything manually.

---

## Troubleshooting

**"Module not found" error** — run `pip install <module-name>` and restart the kernel.

**Notebook takes a long time on the tuning step** — the GridSearchCV cells (Decision Tree, SVM) search over many parameter combinations. This is expected and takes 2–5 minutes depending on your machine.

**ILP notebook errors** — make sure SWI-Prolog is installed and accessible from your terminal (`swipl --version` should print a version number).

**Paths not resolving** — make sure you launch Jupyter from the project root directory (`mlds-group5-project/`), not from inside a subfolder.
