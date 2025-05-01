# Replication of Gu, Kelly, and Xiu (2020): *Empirical Asset Pricing via Machine Learning*

This repository replicates and extends the data preparation and modeling pipeline presented in Gu, Kelly, and Xiu (2020). The project builds a large panel of monthly stock-level returns and firm characteristics, and applies a variety of linear and machine learning models to predict excess returns.

## Dataset Overview

- **Time period**: March 1957 to December 2016  
- **Stocks**: ~30,000 unique stocks, ~6,200 on average per month  
- **Target**: Monthly excess returns  
- **Predictors**:
  - 94 firm-level characteristics from the GXZ website (61 annual, 13 quarterly, 20 monthly)
  - 74 industry dummies (based on two-digit SIC codes)
  - 8 macroeconomic variables (Welch and Goyal, 2008):
    - Dividend-price ratio, earnings-price ratio, book-to-market ratio, net equity issuance
    - Treasury-bill rate, term spread, default spread, stock return variance

## Data Sources

- **Tidy Finance (Python)**: Used to download CRSP-based excess return, market capitalization, and lagged market cap  
- **GXZ Data Library**: For firm-level predictors ([link](https://dachxiu.chicagobooth.edu/#data))  
- **WRDS/CRSP**: Required if replicating the full pipeline manually  
- **Welch and Goyal (2008)**: For macroeconomic variables

## Repository Structure

- `data_construction_notebook.ipynb`: Builds or loads the predictor panel  
- `data_filtering.ipynb`: Optionally filters dataset by removing firms in the bottom 20% of 2016 GDP, reducing computational burden  
- `ols3_final.ipynb`: Estimates benchmark OLS models with and without controls  
- `ols-pls-pcr-enet-random-forest.ipynb`: Fits and compares several linear and ensemble models  
- `train_nn_final.ipynb`: Trains deep neural networks on the panel data  
- `glm3.ipynb`, `simplified_glm.ipynb`, `full_glm.ipynb`: Logistic models for direction prediction  
- `gbrt.ipynb`: Gradient Boosted Regression Trees modeling script

## How to Use This Repository

### Step 1: Build or Load the Dataset

You have two options:

**Option A**: Build from Scratch (requires WRDS and GXZ access)  
- Download firm characteristics from [GXZ Data](https://dachxiu.chicagobooth.edu/#data)  
- Run `data_construction_notebook.ipynb` to fetch CRSP data and merge with predictors  
- This step requires a WRDS username/password

**Option B**: Use Preprocessed `.parquet` Files  
- Clone this repository  
- Set the correct file paths in `data_construction_notebook.ipynb` Block 2  
- Load the dataset and proceed to modeling

### Step 2: Optional Data Filtering (For Efficiency)

If computational resources are limited:  
- Open and run `data_filtering.ipynb`  
- This notebook removes all firm-months for firms in the bottom 20% of 2016 GDP  
- Resulting data is smaller but preserves the structure and statistical properties

### Step 3: Modeling

Apply different model types as desired:

**Linear Benchmarks**:
- `ols3_final.ipynb`: Standard OLS with or without control variables

**Dimensionality Reduction + Regularization**:
- `ols-pls-pcr-enet-random-forest.ipynb`: PLS, PCR, Elastic Net, Random Forest

**Neural Networks**:
- `train_nn_final.ipynb`: Fully connected deep learning model

**Probabilistic Models (Direction of Return)**:
- `glm3.ipynb`, `simplified_glm.ipynb`, `full_glm.ipynb`: Logistic models

**Tree-Based Models**:
- `gbrt.ipynb`: Gradient Boosted Regression Trees

## Dependencies

Install required Python packages using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels wrds \
            tensorflow lightgbm tidyfinance tqdm sqlite3
