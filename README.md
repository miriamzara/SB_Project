
## Environment Setup

To avoid package version conflicts, we recommend creating a new conda environment using the provided environment.yml file:

````{bash}
conda env create -f environment.yml
conda activate SBenv
```

After activating the environment, verify the key package versions:

```{bash}
python -c "import sklearn, numpy, torch; print('scikit-learn:', sklearn.__version__); print('numpy:', numpy.__version__); print('torch:', torch.__version__)"
````

The pretrained models were tested with:

```
scikit-learn 1.6.1
numpy 1.26.4
torch 2.2.2
````

These versions are important because the pretrained scikit-learn pipelines are loaded from .pkl files and may not be compatible with newer scikit-learn versions.

## Preparing the Feature Extraction Files

Unzip classification_ring.zip. Inside the extracted classification_ring/ directory, create a folder named data:

```{bash}
mkdir -p classification_ring/data
````

Then unzip features_ring.zip into that folder, so the final structure is:

```
classification_ring/
└── data/
    └── features_ring/
````

## DSSP Installation

The feature extraction pipeline requires DSSP.

On Linux or Google Colab, install DSSP with:

```{bash}
sudo apt-get update
sudo apt-get install -y dssp
```

Then check the path to the DSSP executable:

```
which mkdssp
```

Add this path to classification_ring/configuration.json under the dssp_file field.

Example:

{
  "dssp_file": "/usr/bin/mkdssp"
}

On macOS, DSSP can usually be installed with conda:

```
conda install -c conda-forge dssp
````

Then verify the executable path:

```
which mkdssp
````

## Software Predictor Usage

This repository includes a command-line predictor that runs pretrained models on a protein structure file.

From the project directory, run:

```
python predict_contacts.py path/to/protein.cif --model model_name --output predictions.tsv
````

Supported model names are:

- logistic
- xgboost
- random_forest

Example:

```
python predict_contacts.py test_data/2f4k.cif --model xgboost --output predictions.tsv
````

Random Forest Model

The pretrained random forest model is too large to store directly in the Git repository.

Download it from the provided Google Drive folder:

https://drive.google.com/drive/folders/1NV1EvYlC4WVzZSrIkM5nfA1W9cMiBv50?usp=sharing

After downloading, rename the .pkl file to:

```
random_forest_model.pkl
````

Then place it in:

```
models/random_forest/
````

The final path should be:

```
models/random_forest/random_forest_model.pkl
````

Command-Line Options

input_structure

Path to a valid protein coordinate file in .pdb or .cif format.

--model

Model backend to use for prediction.

Available options:

logistic
random_forest
xgboost

--output

Path to the output prediction file.

Example:

predictions.tsv

--config

Path to the configuration file. Defaults to:

classification_ring/configuration.json

Training

The training pipeline is reproducible using the provided Jupyter notebooks.

Step 1: Preprocessing

Run:

1_preprocessing.ipynb

This creates a .parquet dataframe from the raw feature data located in:

data/features_ring/

Step 2: Train/Validation/Test Split

Run:

2_train_val_test_split.ipynb

This creates the train, validation, and test splits used by the training notebooks.

Step 3: Model Training and Validation

Run the corresponding notebooks for each model:

3_logistic.ipynb
3B_logistic_validation.ipynb
4_random_forest.ipynb
4B_random_forest_validation.ipynb
7_xgboost.ipynb
7B_xgboost_validation.ipynb

These notebooks train the models and evaluate them on the validation set.

Additional Analyses

To reproduce the feature ablation study for the random forest model, run:

5_feature_ablation.ipynb

To perform cross-validation, run:

8_cross_validation.ipynb

Troubleshooting

scikit-learn model loading error

If you see an error such as:

AttributeError: Can't get attribute '_RemainderColsList'

or:

InconsistentVersionWarning: Trying to unpickle estimator ... from version 1.6.1

make sure that scikit-learn 1.6.1 is installed:

conda install -c conda-forge scikit-learn=1.6.1

PyTorch and NumPy compatibility error

If you see:

RuntimeError: Numpy is not available

make sure NumPy is pinned below version 2:

conda install numpy=1.26.4

Then test:

python -c "import numpy, torch; print(numpy.__version__); print(torch.tensor([1.0, 2.0]).numpy())"