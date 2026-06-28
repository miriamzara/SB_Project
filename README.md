## Train and tune models
To start, create a new conda environment:

```{bash}
conda env create -f environment.yml
conda activate SBenv
```
Start with unzipping classification_ring.zip file and making a folder named data in that folder. Unzip features_ring.zip into data. Continue by installing the dssp package directly onto the Colab, Linux:
```{bash}
!sudo apt-get update
!sudo apt-get install -y dssp
```

Run 1_preprocessing.ipynb
Run 2_train_val_test_split.ipynb

Run files 3, 4B and 7 to train models.

Then verify where it was installed so you can put it in your configuration.json as dssp_file:
```{bash}
!which mkdssp
```
## Software Predictor Usage

To run predictions on a protein structure file, run the following command from the project directory:

```bash
python3 predict_contacts.py path/to/protein.cif --model logistic --output predictions.tsv
```

Options:
input_structure: Path to a valid .pdb or .cif coordinate schema.

--model: Select classification engine backend (logistic, random_forest, xgboost).

--config: Path to paths and configuration tracking maps (defaults to classification_ring/configuration.json).

The link to Overleaf is:

[https://www.overleaf.com/9168223453xrjkzknvgnsw#308b22](https://www.overleaf.com/9168223453xrjkzknvgnsw#308b22)
