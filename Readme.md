# NLP Course Environment Setup (Windows + VS Code + Conda)

## Overview

This document describes how to set up a Python environment for NLP and Machine Learning development using:

* Miniconda
* Python 3.12
* Visual Studio Code
* NLTK
* spaCy
* Transformers
* PyTorch
* Scikit-learn
* Jupyter support inside VS Code

---

## Prerequisites

### Install Miniconda

Download and install Miniconda for Windows.

Verify installation:

```bash
conda --version
```

Expected output:

```text
conda 25.x.x
```

---

### Install Visual Studio Code

Install VS Code and the following extensions:

* Python
* Pylance
* Jupyter

Verify installation:

```bash
code --version
```

---

## Create Conda Environment

Create a dedicated environment:

```bash
conda create -n nlp_course_env python=3.12
```

Activate the environment:

```bash
conda activate nlp_course_env
```

Verify:

```bash
python --version
```

Expected:

```text
Python 3.12.x
```

---

## Upgrade Package Tools

```bash
python -m pip install --upgrade pip setuptools wheel
```

---

## Install Core Data Science Packages

```bash
pip install numpy==1.26.4 pandas==2.2.3 matplotlib==3.10.0 scikit-learn==1.6.0 seaborn==0.13.2
```

Verify:

```bash
python -c "import numpy,pandas,matplotlib,sklearn; print('OK')"
```

---

## Install NLP Packages

```bash
pip install nltk==3.9.1 spacy==3.8.3 textblob==0.18.0.post0 vaderSentiment==3.3.2 transformers==4.47.1 gensim==4.3.3
```

Verify:

```bash
python -c "import nltk,spacy,transformers; print('OK')"
```

---

## Install PyTorch

CPU Version:

```bash
pip install torch==2.5.1
```

Verify:

```bash
python -c "import torch; print(torch.__version__)"
```

---

## Install VS Code Notebook Support

Install Jupyter kernel support for VS Code:

```bash
pip install ipykernel
```

Note:

The following command is NOT required for normal VS Code usage:

```bash
python -m ipykernel install --user --name=nlp_course_env
```

VS Code automatically detects Conda environments.

---

## Download NLTK Resources

Start Python:

```bash
python
```

Run:

```python
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
```

Exit:

```python
exit()
```

---

## Install spaCy English Model

Download the English language model:

```bash
python -m spacy download en_core_web_sm
```

Verify:

```bash
python -c "import spacy; nlp=spacy.load('en_core_web_sm'); print('OK')"
```

---

## Open Project in VS Code

Create project folder:

```bash
mkdir D:\ai_ml_app\nlp_course
cd D:\ai_ml_app\nlp_course
```

Open VS Code:

```bash
code .
```

---

## Select Python Interpreter

Inside VS Code:

1. Press `Ctrl + Shift + P`
2. Select **Python: Select Interpreter**
3. Choose:

```text
Python 3.12 (nlp_course_env)
```

Verify by opening a terminal and running:

```bash
python --version
```

---

## Test Installation

Create `test_setup.py`:

```python
import nltk
import pandas as pd
import numpy as np
import spacy
import transformers
import torch

print("NLTK OK")
print("Pandas OK")
print("NumPy OK")
print("spaCy OK")
print("Transformers OK")
print("Torch OK")

nlp = spacy.load("en_core_web_sm")

doc = nlp("Apple is opening a new office in London.")

for ent in doc.ents:
    print(ent.text, ent.label_)
```

Run:

```bash
python test_setup.py
```

Expected output:

```text
NLTK OK
Pandas OK
NumPy OK
spaCy OK
Transformers OK
Torch OK
Apple ORG
London GPE
```

---

## Useful Commands

### Activate Environment

```bash
conda activate nlp_course_env
```

### Deactivate Environment

```bash
conda deactivate
```

### List Environments

```bash
conda env list
```

### Check Installed Packages

```bash
pip list
```

### Export Dependencies

```bash
pip freeze > requirements.txt
```

### Reinstall Dependencies

```bash
pip install -r requirements.txt
```

---

## Troubleshooting

### Check Active Python

```python
import sys
print(sys.executable)
```

Expected path:

```text
...\miniconda3\envs\nlp_course_env\python.exe
```

### Verify spaCy Installation

```bash
python -m spacy validate
```

### Verify Torch Installation

```bash
python -c "import torch; print(torch.__version__)"
```

---

## Recommended Versions

| Component    | Version |
| ------------ | ------- |
| Python       | 3.12    |
| NumPy        | 1.26.4  |
| Pandas       | 2.2.3   |
| Matplotlib   | 3.10.0  |
| Scikit-learn | 1.6.0   |
| NLTK         | 3.9.1   |
| spaCy        | 3.8.3   |
| Transformers | 4.47.1  |
| PyTorch      | 2.5.1   |

This configuration is stable and well-supported for NLP, machine learning, and data science development on Windows using VS Code.