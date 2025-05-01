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

- `create_data.ipynb`: Builds the full dataset and exports yearly `.parquet` files used by all modeling notebooks  
- `gbrt.ipynb`: Implements Gradient Boosted Regression Trees  
- `glm-3_simplified-glm_full-glm.ipynb`: Contains compiled GLM-based models for return direction prediction  
- `ols-pls-pcr-enet-random-forest.ipynb`: Runs OLS, PLS, PCR, Elastic Net, and Random Forest models  
- `ols3.ipynb`: Runs benchmark OLS models with and without control variables  
- `train_nn_final.ipynb`: Trains deep neural networks on the panel dataset

## How to Use This Repository

### Step 1: Build or Load the Dataset

You have two options:

**Option A**: Build from Scratch (requires WRDS and GXZ access)  
- Download firm characteristics from [GXZ Data](https://dachxiu.chicagobooth.edu/#data)  
- Run `create_data.ipynb` to construct the full panel of predictors and export yearly `.parquet` files  
- This step requires WRDS credentials for CRSP access

**Option B**: Use Prebuilt `.parquet` Files  
- Clone this repository  
- Set the appropriate path in any model notebook to point to the stored `.parquet` files  
- Load the data and begin modeling

### Step 2: Optional Data Filtering (For Efficiency)

If computational resources are limited:  
- Use the GDP-based filter inside your own workflow (refer to legacy `data_filtering.ipynb` if needed)  
- This filter removes all firm-months associated with the bottom 20% of 2016 GDP-ranked firms

### Step 3: Modeling

Apply different model types as desired:

**Linear Benchmarks**:
- `ols3.ipynb`: Standard OLS with 3 predictors

**Dimensionality Reduction + Regularization**:
- `ols-pls-pcr-enet-random-forest.ipynb`: PLS, PCR, Elastic Net, Random Forest

**Neural Networks**:
- `train_nn_final.ipynb`: Fully connected deep learning model

**Probabilistic Models (Direction of Return)**:
- `glm-3_simplified-glm_full-glm.ipynb`: Logistic models including basic, simplified, and full GLMs

**Tree-Based Models**:
- `gbrt.ipynb`: Gradient Boosted Regression Trees

## Dependencies

Install required Python packages using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels wrds \
            tensorflow lightgbm tidyfinance tqdm sqlite3
