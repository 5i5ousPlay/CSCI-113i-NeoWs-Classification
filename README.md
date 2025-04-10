# **CSCI 113i NeoWs Classification**
This repository is created as the final requirement for **CSCI 113i - L 2025**.  
It demonstrates the **CRISP-DM** process with the goal of building a model capable of predicting hazardous asteroids using data from **NASA's Near-Earth Object Web Service (NeoWs) API**.  

This README provides light documentation for the code and instructions for running the ETL pipeline and models.

---

## **Extract, Transform, Load (ETL)**
The **ETL process** is implemented in `etl.py` and is responsible for:
- **Extracting** raw asteroid data from the NASA NeoWs API.
- **Transforming** the data by flattening nested JSON structures and performing basic preprocessing.
- **Loading** the processed data into tabular `.csv` files for further analysis.  

### **Usage**
To use the ETL class, initialize and run it with:
```python
from etl import ETL

API_KEY = "<YOUR NASA API KEY>"
etl = ETL(API_KEY)
etl.run(0, 999)  # Extracts pages 0 to 999 from the API
```
By default, extracted and processed data will be stored in **`./data/`**.

### **Important Notes**
🚨 **API Rate Limit:** The NASA NeoWs API has a limit of **1000 requests per hour**.  
Requesting more than 1000 pages may **result in failed or incomplete extractions**.

---

## **Running Partial ETL Steps**
Given that extraction and transformation can be time-consuming, the ETL process allows for **partial execution**.

### **Extract-Only Mode**
To extract data **without transforming or loading**, use:
```python
etl.run(0, 9, extract_only=True)
```
This will retrieve raw JSON data and save it without further processing.

### **Skip Extraction & Use Pre-Extracted Data**
If extraction has already been completed, you can **skip extraction** and proceed directly to transformation and loading:
```python
etl.run(0, 1000, skip_extract=True, 
        unprocessed_output_file="data/unprocessed/extracted_0_1000.json")
```
*The page range should still be specified for naming consistency.*

---

## **Data Preprocessing**
🚧 *To be documented...*  
Preprocessing steps will include:
- Handling missing values  
- Feature engineering  
- Scaling for neural networks (not required for XGBoost)

---

## **Modeling**

### **Tuner Class**

The `XGBModelTuner` class in `tuner.py` provides a reusable and extensible way to perform hyperparameter optimization for any XGBoost model (e.g., `XGBClassifier`, `XGBRegressor`, `XGBRFClassifier`). It wraps around scikit-learn’s `RandomizedSearchCV` (or any compatible search optimizer) and automates the process of selecting the best model and saving the results.

#### **Features:**
- Supports both classification and regression XGBoost models.
- Accepts any compatible optimizer (`RandomizedSearchCV` by default).
- Allows optional saving of the best model and parameters.
- Easily configurable with cross-validation settings, verbosity, and parallel jobs.

### **Usage**

```python
from sklearn.model_selection import StratifiedKFold
from tuner import XGBModelTuner
import xgboost as xgb

# Define your cross-validation strategy
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize the model
model = xgb.XGBRFClassifier(tree_method='hist', device='cpu', seed=42)

# Define the hyperparameter search space
param_grid = {
    "colsample_bynode": [0.4, 0.6, 0.8],
    "subsample": [0.4, 0.6, 0.8],
    "max_depth": [3, 5, 7],
    "n_estimators": [100, 200, 300, 400],
    "reg_lambda": [1, 1.5, 2],
    "gamma": [0, 0.1, 0.3],
}

# Initialize the tuner
tuner = XGBModelTuner(model, param_grid, X_train, Y_train)

# Run the tuning process
optimized_model = tuner.run(
    save_model=True,              # Save the best model to disk
    save_params=True,             # Save the best hyperparameters to disk
    model_path='xgb_model.json',  # Optional: specify path for model
    param_path='model_params.json',  # Optional: specify path for params
    cv=skf                        # Use stratified K-fold cross-validation
)
```

The best model will be returned and saved (if specified), and the optimal parameters will be printed to the console and optionally saved as a JSON file.

---

### **Gradient Boosted Tree Model**
🚧 *To be documented...*

### **Neural Network Model**
🚧 *To be documented...*

---