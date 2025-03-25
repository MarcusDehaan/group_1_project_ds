# Group 1 Data Science Project - Seminar 2025

## Overview

This project replicates the empirical asset pricing study **"Machine Learning in Empirical Asset Pricing"** by Gu, Kelly, and Xiu (2020, Review of Financial Studies). It focuses on applying and comparing machine learning techniques in empirical asset pricing problems, specifically using stock-level data and predictive characteristics.

## Data Preparation

The dataset is composed of **209 predictive firm-level characteristics** in wide format, signed so that future mean returns increase with the characteristics.

- **Signed Predictors Dataset**: 1.6 GB zipped CSV containing 209 predictive characteristics
- **Missing Data**: Omits Price, Size, and STReversal, which can be downloaded from CRSP.
- The code to automate the download process is available on the website.

### Where to Download the Data

Download the signed predictors dataset from [here](https://www.openassetpricing.com/data/), under **Featured Stock-level Signal Datasets**.

1. **File**: 1.6 GB zipped CSV
2. **Additional Notes**: The dataset omits Price, Size, and STReversal, which are available from CRSP.

---

## How to Use This Notebook

You can use the `dataset_creation_script.ipynb` notebook in two ways:

### Option 1: Use Existing `.parquet` Files

1. Clone this repository.
2. Install required Python packages:
   ```bash
   pip install pandas numpy scikit-learn pyarrow
   ```
3. Open the notebook and skip to **Block 2** to load and test the prepared dataset.

### Option 2: Rebuild the Dataset (Requires WRDS)

1. Get access to [WRDS](https://wrds-www.wharton.upenn.edu/) (Wharton Research Data Services).
2. Download the signed predictors dataset from [here](https://www.openassetpricing.com/data/), under **Featured Stock-level Signal Datasets**:
   - **209 predictive firm-level characteristics** in wide format, signed so future mean returns increase in characteristics (1.6 GB zipped CSV).
   - Omits **Price**, **Size**, and **STReversal**, which can be downloaded from CRSP.
   - Code to automate the download is available on the website.
3. Install required Python packages:
   ```bash
   pip install pandas numpy scikit-learn pyarrow wrds
   ```
4. Open the notebook and run **Block 1** to build the dataset, then run **Block 2** to test the dataset.

---

## Required Python Packages

To ensure the notebook runs correctly, please install the following packages:

```bash
pip install pandas numpy scikit-learn pyarrow wrds
```

---

## Credits

This repository and project are based on the paper:

- **Gu, Kelly, and Xiu (2020)**, "Machine Learning in Empirical Asset Pricing", *Review of Financial Studies*.

Please cite the paper if you are using this data in your work.