# Luxury Fashion Resale Price Prediction

## Project Description

This project performs **machine learning-based price prediction** for luxury fashion resale items using the Vestiaire Collective dataset. The analysis pipeline includes comprehensive data preprocessing, feature engineering, and comparison of linear (Lasso) and non-linear (Random Forest) regression models to predict resale prices.

**Key Features:**

- Data cleaning and feature selection from 900K+ luxury items
- Advanced feature engineering (PCA, log transformations)
- Model comparison: Lasso Regression vs Random Forest
- Feature importance analysis revealing brand hierarchy in resale value

## How to Run

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Execution Steps

1. **Prepare Data**: Place `vestiaire.csv` in the working directory
2. **Run Notebook**: Execute all cells sequentially in Jupyter Notebook/Lab
3. **Model Training**: The notebook automatically:
   - Cleans and preprocesses 900K+ records
   - Trains both Lasso and Random Forest models
   - Outputs evaluation metrics and visualizations

### Quick Start

```python
# If running as script, execute in terminal:
jupyter notebook projectCode.ipynb

# Or convert to .py and run:
jupyter nbconvert --to script projectCode.ipynb
python Untitled1.py
```

## Input/Output

### Input

**Required File:** `vestiaire.csv`

- **Rows:** 900,514 luxury resale items
- **Columns (36 original):**
  - Product: `product_type`, `product_condition`, `brand_name`, `product_color`, etc.
  - Seller: `seller_country`, `seller_badge`, `seller_products_sold`, etc.
  - Target: `seller_price` (continuous variable)

### Output

#### 1. Data Preprocessing

- Cleaned dataset: 900,514 rows × 206 features (after one-hot encoding)
- Feature reduction: 36 → 206 engineered features
- Missing value treatment and outlier handling

#### 2. Model Performance

**Lasso Regression (L1):**

- R² Score: **0.4674**
- RMSE: **0.7557** (log scale)
- Best Alpha: **0.000240**

**Random Forest:**

- R² Score: **~0.47** (expected)
- RMSE: **~0.75** (log scale)
- 50 trees, max_depth=10

#### 3. Key Insights

**Top Price Drivers (Feature Importance):**

1. `brand_name_Rolex` → Coefficient: **+3.59**
2. `brand_name_Omega` → Coefficient: **+2.01**
3. `brand_name_Chanel` → Coefficient: **+1.79**
4. `product_type_Silk tie` → Coefficient: **-1.23** (negative impact)

#### 4. Visualizations

- Correlation heatmap (numeric features)
- Actual vs Predicted scatter plots
- Feature importance bar chart

### Sample Output Files

```
📊 Generated During Execution:
├── Correlation heatmap (matplotlib figure)
├── Actual vs Predicted plots (Lasso + RF)
├── Feature importance dataframe
└── Model evaluation metrics (console output)
```

## Key Findings

- **Hard Luxury dominates**: Rolex adds ~$36 to log price (≈ 3600% price premium)
- **Gender disparity**: Women's items have lower average prices than men's
- **Seller credibility matters**: `seller_pass_rate` shows moderate correlation

## Technical Notes

- Uses **log transformation** on price to handle skewed distribution
- PCA reduces 3 seller metrics → 2 principal components
- Handles **13,736 missing values** via median imputation
- Train/Test split: **80/20** (720K training samples)

## Project Structure

```
.
├── projectCode.ipynb          # Main analysis notebook
├── vestiaire.csv             # Input dataset (required)
└── README.md                 # This file
```

## Dependencies

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
