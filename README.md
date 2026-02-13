# Machine Learning Assignment 2
## Adult Income Prediction using Multiple Machine Learning Models

**Student Information:**
- BITS ID: 2025aa05564
- Name: Prajwal Rastogi
- Email: 2025aa05564@wilp.bits-pilani.ac.in
- Date: February 14, 2026

---

## a. Problem Statement

The objective of this assignment is to predict the income level of individuals based on various demographic and socioeconomic features. This is a binary classification problem where we need to determine whether an individual's income is **≤50K** or **>50K** per year.

The problem involves:
- Implementing and comparing multiple machine learning classification models
- Performing comprehensive data exploration and preprocessing
- Evaluating models using appropriate metrics (Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient)
- Analyzing and interpreting model performance to identify the best-performing model

This classification task is important for understanding socioeconomic patterns and can be applied in various domains such as financial planning, policy making, and market research.

---

## b. Dataset Description

**Dataset Name:** Adult Income Dataset (Census Income Dataset)

**Source:** UCI Machine Learning Repository  
**URL:** https://archive.ics.uci.edu/ml/datasets/adult

**Problem Type:** Binary Classification (Income: ≤50K or >50K)

**Dataset Characteristics:**
- **Total Instances:** ~48,842 (Training: 32,561, Test: 16,281)
- **Number of Features:** 14 features (excluding target variable)
- **Target Variable:** Income (Binary: ≤50K or >50K)
- **Missing Values:** Present (handled during preprocessing)

**Feature Description:**

1. **age** (Numeric): Age of the individual
2. **workclass** (Categorical): Type of employment (Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked)
3. **fnlwgt** (Numeric): Final weight (sampling weight)
4. **education** (Categorical): Education level (Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool)
5. **education-num** (Numeric): Numeric representation of education level
6. **marital-status** (Categorical): Marital status (Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse)
7. **occupation** (Categorical): Occupation type (Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces)
8. **relationship** (Categorical): Relationship status (Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried)
9. **race** (Categorical): Race (White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black)
10. **sex** (Categorical): Gender (Female, Male)
11. **capital-gain** (Numeric): Capital gains
12. **capital-loss** (Numeric): Capital losses
13. **hours-per-week** (Numeric): Hours worked per week
14. **native-country** (Categorical): Country of origin

**Preprocessing Steps:**
- Handled missing values (filled numeric columns with median, categorical with mode)
- Encoded categorical variables using Label Encoding
- Performed train-test split (80% training, 20% testing)
- Applied StandardScaler for feature scaling (important for Logistic Regression, KNN, and Naive Bayes)

---

## c. Models Used

Six different machine learning classification models were implemented and evaluated on the Adult Income dataset:

1. **Logistic Regression** - Linear model for binary classification
2. **Decision Tree Classifier** - Non-linear tree-based model
3. **K-Nearest Neighbor (KNN) Classifier** - Instance-based learning algorithm
4. **Naive Bayes Classifier (Gaussian)** - Probabilistic classifier based on Bayes' theorem
5. **Random Forest (Ensemble)** - Ensemble of decision trees using bagging
6. **XGBoost (Ensemble)** - Gradient boosting ensemble method

### Comparison Table with Evaluation Metrics

All models were evaluated using 6 metrics: Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

| ML Model Name | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC Score |
|---------------|----------|-----------|-----------|--------|----------|-----------|
| Logistic Regression | 0.8229 | 0.8492 | 0.8107 | 0.8229 | 0.8088 | 0.4637 |
| Decision Tree | 0.8567 | 0.8978 | 0.8504 | 0.8567 | 0.8479 | 0.5765 |
| kNN | 0.8273 | 0.8516 | 0.8212 | 0.8273 | 0.8235 | 0.5073 |
| Naive Bayes | 0.8068 | 0.8547 | 0.7908 | 0.8068 | 0.7836 | 0.3950 |
| Random Forest (Ensemble) | 0.8610 | 0.9181 | 0.8559 | 0.8610 | 0.8515 | 0.5883 |
| XGBoost (Ensemble) | 0.8775 | 0.9303 | 0.8732 | 0.8775 | 0.8730 | 0.6463 |

**Key Findings:**
- **Best Overall Model:** XGBoost achieved the highest performance across all metrics
- **Best Accuracy:** XGBoost (87.75%)
- **Best AUC Score:** XGBoost (0.9303)
- **Best Precision:** XGBoost (0.8732)
- **Best Recall:** XGBoost (0.8775)
- **Best F1 Score:** XGBoost (0.8730)
- **Best MCC Score:** XGBoost (0.6463)

---

## d. Observations on Model Performance

### Model Performance Analysis

| ML Model Name | Observation about Model Performance |
|---------------|-------------------------------------|
| **Logistic Regression** | Logistic Regression achieved a moderate accuracy of 82.29% with an AUC score of 0.8492. The model shows balanced performance across precision (0.8107) and recall (0.8229), indicating it handles both classes reasonably well. However, the MCC score of 0.4637 suggests moderate correlation between predictions and actual values. The model benefits from feature scaling and works well as a baseline linear model. It is interpretable and computationally efficient, making it suitable for initial analysis. |
| **Decision Tree** | Decision Tree performed well with an accuracy of 85.67% and an impressive AUC score of 0.8978. The model shows good precision (0.8504) and recall (0.8567), indicating it captures patterns effectively. The MCC score of 0.5765 shows a stronger correlation than Logistic Regression. Decision Trees are interpretable and can capture non-linear relationships, but they may be prone to overfitting. The model's performance suggests it learned meaningful decision boundaries from the data. |
| **kNN** | K-Nearest Neighbor achieved an accuracy of 82.73% with an AUC score of 0.8516. The model shows balanced precision (0.8212) and recall (0.8273), with an F1 score of 0.8235. The MCC score of 0.5073 indicates moderate performance. KNN's performance is sensitive to feature scaling, which was applied in this implementation. The model works well for local patterns but can be computationally expensive for large datasets. The performance suggests that similar individuals tend to have similar income levels. |
| **Naive Bayes** | Naive Bayes achieved the lowest accuracy (80.68%) among all models, but interestingly achieved a competitive AUC score of 0.8547. The model shows lower precision (0.7908) and recall (0.8068) compared to other models, with an MCC score of 0.3950. The "naive" assumption of feature independence may not hold well for this dataset, as features like education and occupation are likely correlated. However, the model is fast and requires minimal training time, making it useful for quick baseline comparisons. |
| **Random Forest (Ensemble)** | Random Forest performed excellently with an accuracy of 86.10% and the second-highest AUC score of 0.9181. The model shows strong precision (0.8559) and recall (0.8610), with an MCC score of 0.5883. As an ensemble method, Random Forest reduces overfitting by averaging multiple decision trees. The model captures complex non-linear relationships and feature interactions effectively. The performance demonstrates the power of ensemble methods in improving prediction accuracy and robustness. |
| **XGBoost (Ensemble)** | XGBoost emerged as the best-performing model across all metrics, achieving the highest accuracy (87.75%), AUC score (0.9303), precision (0.8732), recall (0.8775), F1 score (0.8730), and MCC score (0.6463). The gradient boosting approach sequentially improves predictions by learning from previous errors. XGBoost's superior performance can be attributed to its ability to handle complex feature interactions, automatic feature selection, and regularization techniques. The model shows excellent generalization with balanced performance across all evaluation metrics, making it the optimal choice for this classification task. |

### Overall Observations:

1. **Ensemble Methods Dominate:** Both ensemble methods (Random Forest and XGBoost) outperformed individual models, demonstrating the effectiveness of combining multiple weak learners.

2. **XGBoost Superiority:** XGBoost achieved the best performance across all metrics, indicating its effectiveness in handling complex patterns in the Adult Income dataset.

3. **Feature Scaling Impact:** Models that benefit from feature scaling (Logistic Regression, KNN, Naive Bayes) showed improved performance after standardization.

4. **Balanced Performance:** Most models showed balanced precision and recall, indicating they handle both income classes (≤50K and >50K) reasonably well.

5. **MCC Score Insights:** The MCC scores provide additional insights into model performance, with XGBoost showing the strongest correlation (0.6463) between predictions and actual values.

6. **Model Complexity vs. Performance:** More complex models (XGBoost, Random Forest) generally performed better, but at the cost of reduced interpretability compared to simpler models like Logistic Regression.

---

## Conclusion

The comprehensive evaluation of six machine learning models on the Adult Income dataset demonstrates that ensemble methods, particularly XGBoost, provide the best predictive performance. XGBoost achieved an accuracy of 87.75% and excelled across all evaluation metrics, making it the recommended model for this classification task. The study also highlights the importance of proper data preprocessing, feature scaling, and model selection in achieving optimal performance.

---

**Note:** This README content should be included in the submitted PDF file as per assignment requirements.
