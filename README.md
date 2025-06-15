# Antigenic Peptide Predictor

A machine learning project to classify short protein sequences (peptides) as potentially antigenic or non-antigenic. This tool serves as a foundational example of applying classification models to bioinformatics problems.

## Project Overview

The model is trained on a labeled dataset of peptide sequences. It uses a **k-mer frequency approach (character 3-grams)** to convert the textual sequence data into a numerical format suitable for machine learning. A **Random Forest Classifier** is then trained on these features to perform the classification.

The final trained model and the vectorizer are saved using `joblib` for future predictions on new, unseen peptide data.



## How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
2.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn joblib
    ```
3.  **Run the script:**
    ```bash
    python predictor.py
    ```

## Performance

The script will train the model and output the evaluation results on the test set.

**Expected Output:**

--- Model Evaluation Results ---
Model Accuracy: 1.0000

Classification Report:
precision    recall  f1-score   support

Non-Antigenic (0)       1.00      1.00      1.00         5
Antigenic (1)       1.00      1.00      1.00         5

     accuracy                           1.00        10
    macro avg       1.00      1.00      1.00        10
 weighted avg       1.00      1.00      1.00        10
