# Fraud Detector

This project implements a Machine Learning pipeline that classifies credit card transactions as "Normal" or "Fraudulent" using Scikit-learn.

## Results

We compared two models using Cross-Validation to evaluate the balance between Precision and Recall:

The Pipeline with SMOTE + LogisticRegression resulted in an extremely low average Precision of 6.6%, meaning 93% of its alerts were false positives.

The Pipeline with SMOTE + RandomForest returned a better result, with 85.1% Precision and 81.9% Recall.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/ArthurDOli/fraud-detector.git
    cd fraud-detector
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the dataset** at: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

5.  **Move the `creditcard.csv` file** to the `/data/` folder in the project root.

6.  **Project Execution**

After setup, you can run the main pipeline or the tests.

1.  **To run the main pipeline:**

    - This command will load the data, preprocess, train the model (SMOTE + RandomForest) and display the Cross-Validation scores and final test scores in the terminal.

    ```bash
    python src/main.py
    ```

2.  **To run the unit tests:**
    - This command will run all tests (`test_*.py`) located in the `/tests/` folder using `pytest`.
    ```bash
    pytest
    ```

## Project Structure

```bash
/fraud-detector
├── /notebooks/
|   └── fraud.ipynb
├── /src/
|   ├── data_processing.py
|   └── evaluation.py
|   └── main.py
|   └── model_training.py
├── /tests/
|   ├── test_data_processing.py
|   ├── test_evaluation.py
|   ├── test_model_training.py
```

- notebooks/: Contains the Jupyter Notebooks used for initial data analysis and experimentation.
- src/: Contains all the pipeline logic.
  - data_processing.py: Functions to load and preprocess data.
  - model_training.py: Functions to split data (`split_data`) and create model pipelines (e.g., `create_smote_rf_pipeline`).
  - evaluation.py: Functions for evaluation (`get_cv_scores`, `get_final_metrics`).
  - main.py: The orchestrator that calls all functions in the correct order.
- tests/: Contains the unit tests made with PyTest.
