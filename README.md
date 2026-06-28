To start, create a new conda environment:

```{bash}
conda env create -f environment.yml
conda activate SBenv
```
Start with unzipping classification_ring.zip file and making a folder named data in that folder. Unzip features_ring.zip into data. Continue by installing the dssp package directly onto the Colab, Linux:
!sudo apt-get update
!sudo apt-get install -y dssp

Run 1_preprocessing.ipynb

Then verify where it was installed so you can put it in your configuration.json as dssp_file:
!which mkdssp

The link to Overleaf is:

[https://www.overleaf.com/9168223453xrjkzknvgnsw#308b22](https://www.overleaf.com/9168223453xrjkzknvgnsw#308b22)
