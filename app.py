import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm

from scipy.stats import (
    shapiro,
    levene,
    ttest_ind,
    mannwhitneyu,
    f_oneway,
    jarque_bera
)

from statsmodels.stats.stattools import omni_normtest
from statsmodels.stats.outliers_influence import variance_inflation_factor


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Medical Insurance Analysis",
    page_icon="🏥",
    layout="wide"
)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv("data/insurance.csv")


# =========================================================
# CREATE REGRESSION MODEL
# =========================================================

# Convert categorical variables into dummy variables
df_model = pd.get_dummies(
    df,
    columns=["sex", "smoker", "region"],
    drop_first=True
)

# Make sure dummy variables are numeric
for col in df_model.columns:
    if df_model[col].dtype == "bool":
        df_model[col] = df_model[col].astype(int)

# Target variable
y = df_model["charges"]

# Predictor variables
X = df_model.drop("charges", axis=1)

# Add intercept
X = sm.add_constant(X)

# Fit OLS model
model = sm.OLS(y, X).fit()

# Fitted values and residuals
fitted_values = model.fittedvalues
residuals = model.resid


# =========================================================
# TITLE
# =========================================================

st.title("🏥 Medical Insurance Charges Analysis")

st.write(
    "Interactive statistical analysis, hypothesis testing, "
    "regression and prediction dashboard."
)


# =========================================================
# THREE TABS
# =========================================================

tab1, tab2, tab3 = st.tabs([
    "📊 Data Exploration",
    "🧪 Hypothesis Testing Lab",
    "🔮 Live Prediction & Diagnostics"
])


# =========================================================
# TAB 1 — DATA EXPLORATION
# =========================================================

with tab1:

    st.header("📊 Data Exploration")

    st.write(
        "Explore the medical insurance dataset using interactive "
        "filters, summary statistics and visualizations."
    )

    # -----------------------------------------------------
    # SIDEBAR FILTERS
    # -----------------------------------------------------

    st.sidebar.header("🔍 Filters")

    # Age filter
    age_range = st.sidebar.slider(
        "Select Age Range",
        min_value=int(df["age"].min()),
        max_value=int(df["age"].max()),
        value=(
            int(df["age"].min()),
            int(df["age"].max())
        )
    )

    # Charges filter
    charges_range = st.sidebar.slider(
        "Select Charges Range",
        min_value=float(df["charges"].min()),
        max_value=float(df["charges"].max()),
        value=(
            float(df["charges"].min()),
            float(df["charges"].max())
        )
    )

    # Smoker filter
    smoker_options = st.sidebar.multiselect(
        "Smoking Status",
        options=sorted(df["smoker"].unique()),
        default=sorted(df["smoker"].unique())
    )

    # Sex filter
    sex_options = st.sidebar.multiselect(
        "Sex",
        options=sorted(df["sex"].unique()),
        default=sorted(df["sex"].unique())
    )

    # Region filter
    region_options = st.sidebar.multiselect(
        "Region",
        options=sorted(df["region"].unique()),
        default=sorted(df["region"].unique())
    )

    # -----------------------------------------------------
    # APPLY FILTERS
    # -----------------------------------------------------

    filtered_df = df[
        (df["age"] >= age_range[0]) &
        (df["age"] <= age_range[1]) &
        (df["charges"] >= charges_range[0]) &
        (df["charges"] <= charges_range[1]) &
        (df["smoker"].isin(smoker_options)) &
        (df["sex"].isin(sex_options)) &
        (df["region"].isin(region_options))
    ]

    # -----------------------------------------------------
    # DATASET INFORMATION
    # -----------------------------------------------------

    st.subheader("Filtered Dataset")

    st.write(
        f"Number of records: **{len(filtered_df)}**"
    )

    st.dataframe(
        filtered_df,
        use_container_width=True
    )

    # -----------------------------------------------------
    # SUMMARY STATISTICS
    # -----------------------------------------------------

    st.subheader("Summary Statistics")

    numerical_columns = [
        "age",
        "bmi",
        "children",
        "charges"
    ]

    summary = filtered_df[numerical_columns].describe().T

    st.dataframe(
        summary.round(2),
        use_container_width=True
    )

    # -----------------------------------------------------
    # INTERACTIVE VISUALIZATIONS
    # -----------------------------------------------------

    st.subheader("Interactive Visualizations")

    # Age vs Charges
    st.write("### Age vs Medical Charges")

    fig1 = px.scatter(
        filtered_df,
        x="age",
        y="charges",
        color="smoker",
        hover_data=[
            "bmi",
            "children",
            "region"
        ],
        title="Age vs Medical Charges"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    # BMI vs Charges
    st.write("### BMI vs Medical Charges")

    fig2 = px.scatter(
        filtered_df,
        x="bmi",
        y="charges",
        color="smoker",
        hover_data=[
            "age",
            "children",
            "region"
        ],
        title="BMI vs Medical Charges"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # Charges by Smoker
    st.write("### Medical Charges by Smoking Status")

    fig3 = px.box(
        filtered_df,
        x="smoker",
        y="charges",
        color="smoker",
        title="Medical Charges by Smoking Status"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )


# =========================================================
# TAB 2 — HYPOTHESIS TESTING LAB
# =========================================================

with tab2:

    st.header("🧪 Hypothesis Testing Lab")

    st.write(
        "Select a categorical variable and a numerical variable. "
        "The application automatically selects the appropriate "
        "hypothesis test."
    )

    # -----------------------------------------------------
    # VARIABLE SELECTION
    # -----------------------------------------------------

    categorical_variable = st.selectbox(
        "Select Categorical Variable",
        [
            "smoker",
            "sex",
            "region"
        ]
    )

    numerical_variable = st.selectbox(
        "Select Numerical Variable",
        [
            "charges",
            "age",
            "bmi",
            "children"
        ]
    )

    alpha = 0.05

    st.write("---")

    st.subheader("Selected Variables")

    st.write(
        f"Categorical Variable: **{categorical_variable}**"
    )

    st.write(
        f"Numerical Variable: **{numerical_variable}**"
    )

    # Number of groups
    groups = df[categorical_variable].dropna().unique()

    st.write(
        f"Number of groups: **{len(groups)}**"
    )

    # =====================================================
    # TWO-GROUP TEST
    # =====================================================

    if len(groups) == 2:

        st.subheader("Two-Group Hypothesis Test")

        group1 = df[
            df[categorical_variable] == groups[0]
        ][numerical_variable].dropna()

        group2 = df[
            df[categorical_variable] == groups[1]
        ][numerical_variable].dropna()

        # -------------------------------------------------
        # HYPOTHESES
        # -------------------------------------------------

        st.write("### Hypotheses")

        st.write(
            f"**H₀:** The distributions of {numerical_variable} "
            f"are the same for the two groups."
        )

        st.write(
            f"**H₁:** The distributions of {numerical_variable} "
            f"are different for the two groups."
        )

        # -------------------------------------------------
        # SHAPIRO-WILK
        # -------------------------------------------------

        st.write("### 1. Shapiro-Wilk Normality Test")

        shapiro1 = shapiro(group1)
        shapiro2 = shapiro(group2)

        normal1 = shapiro1.pvalue >= alpha
        normal2 = shapiro2.pvalue >= alpha

        col1, col2 = st.columns(2)

        with col1:

            st.write(f"**{groups[0]}**")

            st.write(
                f"Statistic: {shapiro1.statistic:.4f}"
            )

            st.write(
                f"p-value: {shapiro1.pvalue:.6g}"
            )

        with col2:

            st.write(f"**{groups[1]}**")

            st.write(
                f"Statistic: {shapiro2.statistic:.4f}"
            )

            st.write(
                f"p-value: {shapiro2.pvalue:.6g}"
            )

        # -------------------------------------------------
        # LEVENE TEST
        # -------------------------------------------------

        st.write("### 2. Levene's Test for Equal Variances")

        levene_result = levene(group1, group2)

        st.write(
            f"Test statistic: **{levene_result.statistic:.4f}**"
        )

        st.write(
            f"p-value: **{levene_result.pvalue:.6g}**"
        )

        # -------------------------------------------------
        # CHOOSE TEST
        # -------------------------------------------------

        st.write("### 3. Selected Statistical Test")

        if normal1 and normal2:

            st.info(
                "Both groups are approximately normal. "
                "A two-sample t-test is used."
            )

            test_result = ttest_ind(
                group1,
                group2,
                equal_var=(
                    levene_result.pvalue >= alpha
                )
            )

            test_name = "Two-Sample t-Test"

        else:

            st.info(
                "At least one group is not normally distributed. "
                "A Mann-Whitney U test is used."
            )

            test_result = mannwhitneyu(
                group1,
                group2,
                alternative="two-sided"
            )

            test_name = "Mann-Whitney U Test"

        # -------------------------------------------------
        # RESULTS
        # -------------------------------------------------

        st.write("### 4. Test Results")

        st.write(
            f"**Test:** {test_name}"
        )

        st.write(
            f"Test statistic: **{test_result.statistic:.4f}**"
        )

        st.write(
            f"p-value: **{test_result.pvalue:.6g}**"
        )

        # -------------------------------------------------
        # CONCLUSION
        # -------------------------------------------------

        st.write("### 5. Conclusion")

        if test_result.pvalue < alpha:

            st.error(
                "Reject H₀"
            )

            st.write(
                f"There is statistically significant evidence "
                f"that {numerical_variable} differs between "
                f"the two groups."
            )

        else:

            st.success(
                "Fail to Reject H₀"
            )

            st.write(
                f"There is not sufficient statistical evidence "
                f"that {numerical_variable} differs between "
                f"the two groups."
            )

    # =====================================================
    # MULTIPLE-GROUP TEST
    # =====================================================

    else:

        st.subheader("One-Way ANOVA")

        st.write("### Hypotheses")

        st.write(
            f"**H₀:** The mean {numerical_variable} is the same "
            f"across all groups."
        )

        st.write(
            f"**H₁:** At least one group has a different mean "
            f"{numerical_variable}."
        )

        # Create groups
        group_data = []

        for group in groups:

            values = df[
                df[categorical_variable] == group
            ][numerical_variable].dropna()

            group_data.append(values)

        # ANOVA
        anova_result = f_oneway(*group_data)

        st.write("### ANOVA Results")

        st.write(
            f"F-statistic: **{anova_result.statistic:.4f}**"
        )

        st.write(
            f"p-value: **{anova_result.pvalue:.6g}**"
        )

        # -------------------------------------------------
        # GROUP MEANS
        # -------------------------------------------------

        st.write("### Group Means")

        group_means = (
            df.groupby(categorical_variable)[numerical_variable]
            .mean()
            .sort_values(ascending=False)
        )

        st.dataframe(
            group_means.round(2),
            use_container_width=True
        )

        # -------------------------------------------------
        # CONCLUSION
        # -------------------------------------------------

        st.write("### Conclusion")

        if anova_result.pvalue < alpha:

            st.error(
                "Reject H₀"
            )

            st.write(
                f"There is statistically significant evidence "
                f"that mean {numerical_variable} differs across "
                f"the groups."
            )

        else:

            st.success(
                "Fail to Reject H₀"
            )

            st.write(
                f"There is not sufficient statistical evidence "
                f"that mean {numerical_variable} differs across "
                f"the groups."
            )


# =========================================================
# TAB 3 — LIVE PREDICTION & DIAGNOSTICS
# =========================================================

with tab3:

    st.header("🔮 Live Prediction & Diagnostics")

    st.write(
        "Enter patient information to generate a real-time "
        "medical charge prediction."
    )

    # -----------------------------------------------------
    # USER INPUTS
    # -----------------------------------------------------

    st.subheader("Patient Information")

    col1, col2 = st.columns(2)

    with col1:

        input_age = st.number_input(
            "Age",
            min_value=int(df["age"].min()),
            max_value=int(df["age"].max()),
            value=30
        )

        input_bmi = st.number_input(
            "BMI",
            min_value=float(df["bmi"].min()),
            max_value=float(df["bmi"].max()),
            value=30.0
        )

        input_children = st.number_input(
            "Number of Children",
            min_value=int(df["children"].min()),
            max_value=int(df["children"].max()),
            value=0,
            step=1
        )

    with col2:

        input_sex = st.selectbox(
            "Sex",
            sorted(df["sex"].unique())
        )

        input_smoker = st.selectbox(
            "Smoking Status",
            sorted(df["smoker"].unique())
        )

        input_region = st.selectbox(
            "Region",
            sorted(df["region"].unique())
        )

    # -----------------------------------------------------
    # CREATE INPUT DATAFRAME
    # -----------------------------------------------------

    input_data = pd.DataFrame({
        "age": [input_age],
        "bmi": [input_bmi],
        "children": [input_children],
        "sex": [input_sex],
        "smoker": [input_smoker],
        "region": [input_region]
    })

    # Encode input in same way as training data
    input_encoded = pd.get_dummies(
        input_data,
        columns=["sex", "smoker", "region"],
        drop_first=True
    )

    # Make sure all model columns exist
    input_encoded = input_encoded.reindex(
        columns=X.columns.drop("const"),
        fill_value=0
    )

    # Add constant
    input_encoded = sm.add_constant(
        input_encoded,
        has_constant="add"
    )

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    prediction_result = model.get_prediction(
        input_encoded
    )

    prediction_summary = prediction_result.summary_frame(
        alpha=0.05
    )

    predicted_charge = prediction_summary[
        "mean"
    ].iloc[0]

    mean_ci_lower = prediction_summary[
        "mean_ci_lower"
    ].iloc[0]

    mean_ci_upper = prediction_summary[
        "mean_ci_upper"
    ].iloc[0]

    obs_ci_lower = prediction_summary[
        "obs_ci_lower"
    ].iloc[0]

    obs_ci_upper = prediction_summary[
        "obs_ci_upper"
    ].iloc[0]

    # -----------------------------------------------------
    # DISPLAY PREDICTION
    # -----------------------------------------------------

    st.subheader("Predicted Medical Charges")

    st.metric(
        "Predicted Charges",
        f"${predicted_charge:,.2f}"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.write("### 95% Confidence Interval")

        st.write(
            f"${mean_ci_lower:,.2f} "
            f"to "
            f"${mean_ci_upper:,.2f}"
        )

    with col2:

        st.write("### 95% Prediction Interval")

        st.write(
            f"${obs_ci_lower:,.2f} "
            f"to "
            f"${obs_ci_upper:,.2f}"
        )

    st.info(
        "The confidence interval describes uncertainty around "
        "the estimated mean charge, while the prediction interval "
        "is wider and represents the range for an individual case."
    )

    # =====================================================
    # REGRESSION DIAGNOSTICS
    # =====================================================

    st.write("---")

    st.subheader("Regression Diagnostics")

    # -----------------------------------------------------
    # RESIDUALS VS FITTED
    # -----------------------------------------------------

    st.write("### Residuals vs Fitted Values")

    residual_df = pd.DataFrame({
        "Fitted Values": fitted_values,
        "Residuals": residuals
    })

    fig_residual = px.scatter(
        residual_df,
        x="Fitted Values",
        y="Residuals",
        title="Residuals vs Fitted Values"
    )

    fig_residual.add_hline(
        y=0,
        line_dash="dash"
    )

    st.plotly_chart(
        fig_residual,
        use_container_width=True
    )

    # -----------------------------------------------------
    # Q-Q PLOT
    # -----------------------------------------------------

    st.write("### Q-Q Plot")

    from scipy.stats import probplot

    theoretical_quantiles, ordered_values = probplot(
        residuals,
        dist="norm"
    )

    qq_df = pd.DataFrame({
        "Theoretical Quantiles": theoretical_quantiles[0],
        "Ordered Residuals": theoretical_quantiles[1]
    })

    fig_qq = px.scatter(
        qq_df,
        x="Theoretical Quantiles",
        y="Ordered Residuals",
        title="Q-Q Plot of Residuals"
    )

    # Add approximate reference line
    slope, intercept = np.polyfit(
        theoretical_quantiles[0],
        theoretical_quantiles[1],
        1
    )

    x_line = np.array([
        theoretical_quantiles[0].min(),
        theoretical_quantiles[0].max()
    ])

    y_line = intercept + slope * x_line

    fig_qq.add_scatter(
        x=x_line,
        y=y_line,
        mode="lines",
        name="Reference Line"
    )

    st.plotly_chart(
        fig_qq,
        use_container_width=True
    )

    # -----------------------------------------------------
    # JARQUE-BERA TEST
    # -----------------------------------------------------

    st.write("### Jarque-Bera Normality Test")

    jb_result = jarque_bera(residuals)

    st.write(
        f"JB Statistic: **{jb_result.statistic:.4f}**"
    )

    st.write(
        f"p-value: **{jb_result.pvalue:.6g}**"
    )

    if jb_result.pvalue < 0.05:

        st.warning(
            "Reject H₀: The residuals show evidence of "
            "non-normality."
        )

    else:

        st.success(
            "Fail to Reject H₀: There is not sufficient evidence "
            "of non-normal residuals."
        )

    # -----------------------------------------------------
    # OMNIBUS TEST
    # -----------------------------------------------------

    st.write("### Omnibus Normality Test")

    omnibus_stat, omnibus_pvalue = omni_normtest(
        residuals
    )

    st.write(
        f"Test Statistic: **{omnibus_stat:.4f}**"
    )

    st.write(
        f"p-value: **{omnibus_pvalue:.6g}**"
    )

    if omnibus_pvalue < 0.05:

        st.warning(
            "Reject H₀: The residuals show evidence of "
            "non-normality."
        )

    else:

        st.success(
            "Fail to Reject H₀: There is not sufficient evidence "
            "of non-normal residuals."
        )

    # =====================================================
    # VIF
    # =====================================================

    st.write("---")

    st.subheader("Variance Inflation Factor (VIF)")

    continuous_vars = df_model[
        ["age", "bmi", "children"]
    ]

    vif_data = pd.DataFrame()

    vif_data["Variable"] = continuous_vars.columns

    vif_data["VIF"] = [
        variance_inflation_factor(
            continuous_vars.values,
            i
        )
        for i in range(
            continuous_vars.shape[1]
        )
    ]

    st.dataframe(
        vif_data.round(3),
        use_container_width=True
    )

    st.write(
        "VIF values close to 1 indicate very little "
        "multicollinearity. Values above 5 may indicate "
        "a potential concern."
    )