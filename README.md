# Decision Trees & Random Forests – Loan Repayment Prediction
**Course:** ARTI308 – Machine Learning

---

## Project Overview

This project builds binary classification models using **Decision Trees** and **Random Forests** to predict whether a LendingClub borrower will fully repay their loan. Given a set of borrower financial and behavioral features, both models are trained and compared to determine which better identifies high-risk borrowers.

---

## Dataset

**File:** `loan_data.csv`
**Size:** 9,578 rows × 14 columns
**Source:** [LendingClub.com](https://www.lendingclub.com/) — loan data from 2007–2010

| Feature | Description |
|---|---|
| `credit.policy` | 1 if the customer meets LendingClub's credit underwriting criteria, 0 otherwise |
| `purpose` | Purpose of the loan (e.g., debt consolidation, credit card, small business) |
| `int.rate` | Interest rate of the loan as a proportion |
| `installment` | Monthly installment owed by the borrower if the loan is funded |
| `log.annual.inc` | Natural log of the borrower's self-reported annual income |
| `dti` | Debt-to-income ratio of the borrower |
| `fico` | FICO credit score of the borrower |
| `days.with.cr.line` | Number of days the borrower has had a credit line |
| `revol.bal` | Borrower's revolving balance (unpaid amount at end of billing cycle) |
| `revol.util` | Revolving line utilization rate (credit used relative to total available) |
| `inq.last.6mths` | Number of creditor inquiries in the last 6 months |
| `delinq.2yrs` | Number of times the borrower was 30+ days past due in the past 2 years |
| `pub.rec` | Number of derogatory public records (bankruptcies, tax liens, judgments) |
| `not.fully.paid` | **Target variable** – 1 (did not fully repay) or 0 (fully repaid) |

---

## Techniques Applied

### Task 1 – Data Loading & Exploration
- Loaded the dataset using `pandas`
- Used `.info()` and `.describe()` to examine data types, null values, and summary statistics

### Task 2 – Exploratory Data Analysis (EDA)
- **FICO Histogram by credit.policy** – Distribution of FICO scores split by whether the borrower met LendingClub's underwriting criteria
- **FICO Histogram by not.fully.paid** – Distribution of FICO scores split by loan repayment outcome
- **Countplot by purpose** – Loan counts across each purpose category, colored by repayment status
- **FICO vs. Interest Rate Scatter** – Relationship between credit score and assigned interest rate, colored by repayment outcome

### Task 3 – Data Preprocessing
- Identified `purpose` as a categorical feature requiring encoding
- Applied `pd.get_dummies()` with `drop_first=True` to convert `purpose` into dummy variables, producing `final_data`
- Split data into training (70%) and testing (30%) sets using `train_test_split` with `random_state=101`

### Task 4 – Model Training
- **Decision Tree:** Trained a `DecisionTreeClassifier` with default hyperparameters
- **Random Forest:** Trained a `RandomForestClassifier` with `n_estimators=200` and `random_state=101`

### Task 5 – Evaluation
- Generated **Confusion Matrices** for both models to compare true/false positives and negatives
- Generated **Classification Reports** showing precision, recall, F1-score, and accuracy per class
- Plotted **Feature Importances** from the Random Forest to identify the most predictive variables

---

## Results

| Model | Overall Accuracy | Not-Fully-Paid Recall | Not-Fully-Paid Precision |
|---|---|---|---|
| Decision Tree | 73% | 24% | 19% |
| Random Forest | 85% | 2% | 45% |

---

## Conclusion

Both models were trained to classify whether a LendingClub borrower would fully repay their loan. The EDA revealed that higher FICO scores are associated with lower interest rates and better repayment outcomes, and that debt consolidation is the most common loan purpose. The Random Forest achieved higher overall accuracy (85% vs. 73%), but the dataset is class-imbalanced — only ~16% of loans were not fully repaid — causing the model to predict "fully repaid" almost every time and miss most bad loans (recall of only 2%). The Decision Tree, despite lower overall accuracy, detected more non-repayment cases (recall of 24%). The top predictive features identified by the Random Forest were `int.rate`, `fico`, `installment`, `log.annual.inc`, and `days.with.cr.line`. To improve minority-class detection, future work could explore `class_weight='balanced'`, SMOTE oversampling, or decision threshold tuning.
