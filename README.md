#Signature Extraction Evaluation Framework (SEEF)

SEEF is a framework that can be used to compare different extraction method for mutational signatures, it can handle both the pre and post process, in the pre-process it can generate synthetic data to be used to evaulate with, or just use real data, and in the post-process it will evaulate how good the extraction method did, by comparing the latent result against known signatures,

## Table of Contents

- [1 Usage](#1-usage)
- [2 Installation](#2-installation)
- [3 Running](#3-running)
- [4 Example](#4-example)
- [5 License](#5-license)
- [6 Contributers](#6-contributers)

## 1 Usage
### 1.1 Requirements
- python >= 3.10
- pandas >=1.5.3
- scikit-learn >=1.3.0 
- SigProfilerAssignment >=0.0.32
- matplotlib >=3.8.2
- numpy >=1.24.0
- keras >=3.0.1
- torch >=2.0.1
- scipy >= 1.10


## 2 Installation
### 2.1 Code
to download the code do
```shell
    git clone https://github.com/
```
### 2.2 Known Mutational Signatures
To get the signatures go to
```shell
    https://cancer.sanger.ac.uk/signatures/downloads/
```

and download the file belonging to **GRCh37** and **SBS** as we work with Single-base substitutions

### 2.3 Real data
To get the real data do
```shell
    wget https://dcc.icgc.org/api/v1/download?fn=/PCAWG/mutational_signatures/Input_Data_PCAWG7_23K_Spectra_DB/Mutation_Catalogs_--_Spectra_of_Individual_Tumours/WGS_PCAWG_2018_02_09.zip
```
unzip it



## 3 Running

### 3.1 Creating synthetic dataset
creating synthetic dataset you need to use the function `create_dataset` from the file `synthetic_dataset.py` in folder `sigGen`, there is 3 parameter required for the function to run, 
1. number of signature
2. number of how many samples
3. path to Known Mutational Signatures file

example

```python
create_dataset(5,5,"path/to/sig.txt")
```

### 3.2 Extraction Method 
when creating a Extraction Method there is some requirement both the input and output parameters for it to make it work for this framework

**input**
1. pandas dataframe
2. componets/latent

**output**
1. latents space
2. weights
3. loss

### 3.3 Clustering
For using the clustering method you need to give the class the path to the dataset and a pointer to the function

example
```python
mk_cluster("path/to/dataset",nmf).run()
```
*note that there isn't paranthesis after nmf as it is function*

### 3.4 Method Evaluation
to use this class it needs to imported from the folder `eval` file `method_evaluation.py` and
Depending on what type of dataset one used, changes what function to call from the class

**Synthetic data**
For synthetic data one needs to give 4 file path those are
1. file path to cluster signatures
2. file path to cluster weights
3. file path to known signatures
4. file path to known weights

example
```python
evaluator = MethodEvaluator()
results = evaluator.evaluate(
    "path/to/clusterd_singatures",
    "path/to/clusterd_weights",
    "path/to/known_signatures",
    "path/to/known_weights")
```
**Real data**
For real data only path to the cluster signatures file is needed and which GRCh was used

```python
evaluator = MethodEvaluator()
results = evaluator.COSMICevaluate("path/to/extract_sig",GRCh ="GRCh37")
```

## 4 Example
An example on how to run it with synthetic data
```python
from sigGen.synthetic_dataset import create_dataset
from eval.method_evaluation import MethodEvaluator
from eval.cluster import mk_cluster
from sklearn.decomposition import NMF

def nmf(df, components):
    model = NMF(n_components=components, init="random", random_state=0)
    W = model.fit_transform(df)
    H = model.components_
    return W, H, model.reconstruction_err_


if __name__ == "__main__":
    create_dataset(5,1000,"path/known/signatures.txt")
    mk_cluster("path/to/synthetic_data",nmf).run()
    evaluator = MethodEvaluator()
    results = evaluator.evaluate(
    "path/to/clusterd_singatures",
    "path/to/clusterd_weights",
    "path/to/known_signatures",
    "path/to/known_weights")
    print(results)
```

## 5 License

This project is licensed under the [MIT License](LICENSE). See the `LICENSE` file for more information.

## 6 Contributers

Casper Gislum - Cgislu19@student.aau.dk

Kevin Risgaard Sinding - Ksindi19@student.aau.dk

Magni Jógvansson Hansen - Mjha19@student.aau.dk

Frederik Rasmussen - Frasm19@student.aau.dk

Nikolai Eriksen Kure - Nkure19@student.aau.dk

Mathias Vestergaard Jensen  - Matjen19@student.aau.dk



