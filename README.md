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
### **Gradient Boosted Tree Model**
🚧 *To be documented...*  

### **Neural Network Model**
🚧 *To be documented...*  

---