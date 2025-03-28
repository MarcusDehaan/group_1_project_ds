# 📊 Group 1 Project – GXZ Dataset Construction & Analysis

This project replicates and extends the data preparation process used in the Gu, Kelly, and Xiu (2020) paper, *Empirical Asset Pricing via Machine Learning*.

We use CRSP data from WRDS, and firm characteristics from [Dacheng Xiu’s GXZ data site](https://dachxiu.chicagobooth.edu/) to build a monthly panel of equity-level predictors.

---

## 📁 What's in This Repo

- `dataset_yearly_parquet/`: Cleaned, standardized panel data split by year (`1957–2021`)
- `data_construction_notebook.ipynb`: 
  - Block 1: Build the dataset manually from WRDS + GXZ predictors
  - Block 2: Load prebuilt `.parquet` files + run diagnostics

---

## 🔨 How to Use

### Option 1: 🛠️ Build Dataset from Scratch (WRDS Access Required)
1. Download the GXZ predictors CSV from [here](https://dachxiu.chicagobooth.edu/)
2. Update the CSV path in the notebook
3. Run Block 1 (`wrds.Connection()` + `.raw_sql(...)`) to fetch CRSP and merge

### Option 2: 💾 Use Prebuilt Parquet Files
1. Clone this repo
2. Set `parquet_path` in Block 2 of the notebook
3. Load, validate, and begin modeling!

---

## 📊 Quick Stats
- ✅ Covers: 1957–2021
- ✅ ~105 predictors (GXZ + CRSP)
- ✅ Standardized monthly by cross-section
- ✅ Includes benchmark 3-factor model diagnostics

---

## ⚙️ Dependencies

Install required packages using:

```bash
pip install pandas numpy scikit-learn pyarrow wrds


---

### ✅ To update your README:

```bash
cd path/to/group_1_project_ds
nano README.md  # or open in VS Code or any editor
# paste the new content
git add README.md
git commit -m "Update README with usage and documentation"
git push origin main
