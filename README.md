# Medical Insurance Cost Analysis & Prediction

## Project Overview

This project performs statistical analysis and predictive modeling on medical insurance charges.

The project includes:
- Descriptive statistical analysis
- Data visualization
- Hypothesis testing
- Multiple linear regression
- Regression diagnostics
- Interactive Streamlit dashboard
- Live medical insurance charge prediction

## Dataset

The dataset contains information about medical insurance customers.

### Features

| Feature | Description |
|---|---|
| age | Age of the individual |
| sex | Gender of the individual |
| bmi | Body Mass Index |
| children | Number of children/dependents |
| smoker | Smoking status |
| region | Residential region |
| charges | Medical insurance charges |

The target variable for regression is `charges`.

## Statistical Analysis

### Descriptive Analysis

The numerical variables were analyzed using:

- Mean
- Median
- Standard deviation
- Interquartile range (IQR)
- Skewness
- Kurtosis

Histograms and KDE plots were used to examine distributions.

Scatter plots and correlation matrices were used to study relationships between numerical variables.

## Hypothesis Testing

### Test 1: Smoker vs Medical Charges

The medical charges of smokers and non-smokers were compared.

Because the data did not satisfy the normality assumption, the Mann-Whitney U test was used.

- U statistic = 284133.0
- p-value ≈ 5.27 × 10⁻¹³⁰
- Significance level = 0.05

Since the p-value is less than 0.05, the null hypothesis was rejected.

There is a statistically significant difference in medical charges between smokers and non-smokers.

### Test 2: Region vs Medical Charges

A one-way ANOVA was performed to compare medical charges across the four regions.

- F statistic ≈ 2.970
- p-value ≈ 0.0309
- Significance level = 0.05

Since the p-value is less than 0.05, the null hypothesis was rejected.

This indicates that medical charges differ significantly across regions.

## Multiple Linear Regression

A multiple linear regression model was developed using:

- Age
- BMI
- Number of children
- Sex
- Smoking status
- Region

Categorical variables were converted into dummy variables.

The model achieved:

- R² ≈ 0.751
- Adjusted R² ≈ 0.749

Therefore, the model explains approximately 75.1% of the variation in medical insurance charges.

Smoking status showed the strongest effect on predicted medical charges. Smokers had substantially higher predicted charges than non-smokers after controlling for the other variables.

Age, BMI, number of children, smoking status, Southeast region, and Southwest region were statistically significant predictors at the 5% significance level.

## Streamlit Dashboard

The project includes an interactive Streamlit application with three sections.

### 1. Data Exploration

Users can:
- Filter the dataset
- View summary statistics
- Explore relationships using interactive plots
- Compare charges across groups

### 2. Hypothesis Testing Lab

Users can dynamically select:
- Categorical variables
- Numerical variables

The application automatically performs the appropriate statistical test and displays:
- Test statistic
- p-value
- Decision regarding the null hypothesis

### 3. Live Prediction & Diagnostics

Users can enter:
- Age
- BMI
- Number of children
- Sex
- Smoking status
- Region

The application provides:
- Predicted medical charges
- 95% confidence interval
- 95% prediction interval
- Residual diagnostics
- Q-Q plot
- Normality test results
- VIF values

## Project Structure

```text
Lab4_Statistical_Modeling/
│
├── analysis.ipynb
├── app.py
├── requirements.txt
├── README.md
│
└── data/
    └── insurance.csv