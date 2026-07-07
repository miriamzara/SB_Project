
## Environment Setup

To avoid package version conflicts, we recommend creating a new conda environment using the provided environment.yml file:

```{bash}
conda env create -f environment.yml
conda activate SBenv
```

After activating the environment, verify the key package versions:

```{bash}
python -c "import sklearn, numpy, torch;
print('scikit-learn:', sklearn.__version__);
print('numpy:', numpy.__version__);
print('torch:', torch.__version__)"
```

The pretrained models were tested with:

```{bash}
scikit-learn 1.6.1
numpy 1.26.4
torch 2.2.2
```

These versions are important because the pretrained scikit-learn pipelines are loaded from .pkl files and may not be compatible with newer scikit-learn versions.

## DSSP Installation

The feature extraction pipeline requires DSSP.

On Linux or Google Colab, install DSSP with:

```{bash}
sudo apt-get update
sudo apt-get install -y dssp
```

On macOS, DSSP can usually be installed with conda:

```{bash}
conda install -c conda-forge dssp
````

Then verify the executable path:

```{bash}
which mkdssp
```

# Software Predictor Usage

This repository includes a command-line predictor that runs pretrained models on a protein structure file.

From the project directory, run:

```{bash}
python predict_contacts.py path/to/protein.cif --model model_name --output predictions.tsv
```

Supported model names are:

- logistic
- xgboost
- random_forest

Example:

```{bash}
python predict_contacts.py test_data/2f4k.cif --model xgboost --output predictions.tsv
```

Random Forest Model

The pretrained random forest model is too large to store directly in the Git repository.

Download it from the provided Google Drive folder:

https://drive.google.com/drive/folders/1NV1EvYlC4WVzZSrIkM5nfA1W9cMiBv50?usp=sharing

After downloading, rename the .pkl file to:

```{bash}
random_forest_model.pkl
```

Then place it in:

```{bash}
models/random_forest/
```

The final path should be:

```{bash}
models/random_forest/random_forest_model.pkl
```

## Command-Line Options

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

# Training

The training pipeline is reproducible using the provided Jupyter notebooks.

## Preparing the Feature Extraction Files

You can download our training data at the Google Drive link:

https://drive.google.com/file/d/119BxPujC3rvJxLhnFHE-bwt9SyXPNAZv/view

provided you have rights to access it.

Inside the extracted classification_ring/ directory, if not already present create a folder named data:

```{bash}
mkdir -p classification_ring/data
```

Then unzip features_ring.zip into that folder, so the final structure is:

```{bash}
classification_ring/
└── data/
    └── features_ring/
```


### Step 1: Preprocessing

Run:

```{bash}
1_preprocessing.ipynb
```


This creates a .parquet dataframe from the raw feature data located in:

```{bash}
data/features_ring/
```

### Step 2: Train/Validation/Test Split

Run:

```{bash}
2_train_val_test_split.ipynb
```
This creates the train, validation, and test splits used by the training notebooks.

### Step 3: Model Training and Validation

Run the corresponding notebooks for each model:

```{bash}
3_logistic_regression.ipynb
3B_logistic_biorules.ipynb
4B_random_forest_metrics_added.ipynb
7_XGBoost.ipynb
```

These notebooks train the models and evaluate them on the validation set.

**Additional Analyses**

To reproduce the feature ablation study for the random forest model, run:

```{bash}
5_feature_ablation.ipynb
```

To perform cross-validation, run:

```{bash}
8_cross_validation.ipynb
```
