from pathlib import Path
import pandas as pd


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(data_path: str | Path) -> pd.DataFrame:
    """
    Load and concatenate all CSV files from the raw dataset folder.

    Parameters
    ----------
    data_path : str or Path
        Path to the folder containing raw CSV files.

    Returns
    -------
    pd.DataFrame
        Combined dataset with lowercase column names.
    """

    csv_files = sorted(Path(data_path).glob("*.csv"))

    loan_data = pd.concat(
        [pd.read_csv(file) for file in csv_files],
        ignore_index=True
    )

    loan_data.columns = loan_data.columns.str.lower()

    return loan_data


# =============================================================================
# COLUMN RENAMING
# =============================================================================

def rename_columns(loan_data: pd.DataFrame) -> pd.DataFrame:
    """
    Rename target-related columns for consistency across models.

    Parameters
    ----------
    loan_data : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    loan_data = loan_data.copy()

    loan_data.rename(
        columns={
            "default_flag": "is_currently_default",
            "target_default_12m_flag": "target"
        },
        inplace=True
    )

    return loan_data


# =============================================================================
# DATE CONVERSION
# =============================================================================

def convert_date_columns(loan_data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all *_dt columns to datetime.

    Parameters
    ----------
    loan_data : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    loan_data = loan_data.copy()

    date_cols = [
        col
        for col in loan_data.columns
        if col.endswith("_dt")
    ]

    loan_data[date_cols] = loan_data[date_cols].apply(
        pd.to_datetime,
        errors="coerce"
    )

    return loan_data


# =============================================================================
# DATA QUALITY CHECKS
# =============================================================================

def run_data_quality_checks(loan_data: pd.DataFrame) -> None:
    """
    Run basic business-rule and data-quality checks.

    Parameters
    ----------
    loan_data : pd.DataFrame

    Returns
    -------
    None
    """

    print(f"Duplicate rows: {loan_data.duplicated().sum()}")

    print(
        f"Duplicate account-month rows: "
        f"{loan_data.duplicated(subset=['acct_id', 'snapshot_dt']).sum()}"
    )

    print(
        f"INQ_12M_CNT < INQ_6M_CNT: "
        f"{(loan_data['inq_12m_cnt'] < loan_data['inq_6m_cnt']).sum()}"
    )

    print(
        f"open_trades_cnt > total_trades_cnt: "
        f"{(loan_data['open_trades_cnt'] > loan_data['total_trades_cnt']).sum()}"
    )

    print(
        f"d_dpd_max_12m_cnt < d_dpd_max_6m_cnt: "
        f"{(loan_data['d_dpd_max_12m_cnt'] < loan_data['d_dpd_max_6m_cnt']).sum()}"
    )

    print(
        f"d_times_30dpd_12m_cnt < d_times_30dpd_6m_cnt: "
        f"{(loan_data['d_times_30dpd_12m_cnt'] < loan_data['d_times_30dpd_6m_cnt']).sum()}"
    )

    print(
        f"b_curr_bal_amt > b_max_bal_12m_amt: "
        f"{(loan_data['b_curr_bal_amt'] > loan_data['b_max_bal_12m_amt']).sum()}"
    )

    print(
        f"p_ontime_pay_12m_cnt > 12: "
        f"{(loan_data['p_ontime_pay_12m_cnt'] > 12).sum()}"
    )

    invalid_dates = (
        loan_data["acct_open_dt"] >
        loan_data["snapshot_dt"]
    ).sum()

    print(f"Invalid account dates: {invalid_dates}")


# =============================================================================
# DEVELOPMENT / VALIDATION / OOT SPLIT
# =============================================================================

def split_dev_validation_oot(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into Development, Validation and OOT datasets.

    Parameters
    ----------
    data : pd.DataFrame

    Returns
    -------
    tuple
        (development_data, validation_data, oot_data)
    """

    development_data = data[
        data["dev_oot_flag"] == "DEVELOPMENT"
    ].copy()

    validation_data = data[
        data["dev_oot_flag"] == "VALIDATION"
    ].copy()

    oot_data = data[
        data["dev_oot_flag"] == "OUT_OF_TIME"
    ].copy()

    return development_data, validation_data, oot_data


# =============================================================================
# REPLACE NEVER DELINQUENT VALUE
# =============================================================================

def replace_never_delinquent_value(loan_data: pd.DataFrame) -> pd.DataFrame:
    """
    Replace the sentinel value (999) in d_mob_since_last_delinq_cnt
    with -1 to represent accounts that have never been delinquent.

    Parameters
    ----------
    loan_data : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    loan_data = loan_data.copy()

    loan_data["d_mob_since_last_delinq_cnt"] = (
        loan_data["d_mob_since_last_delinq_cnt"].replace(999, -1)
    )

    return loan_data


def preprocess_lgd(data, pipeline):
    """
    Apply the LGD preprocessing pipeline to new data.
    """

    import pandas as pd

    data = data.copy()

    # unpack pipeline
    encoder = pipeline["encoder"]
    scaler = pipeline["scaler"]

    imputation_values = pipeline["imputation_values"]
    
    categorical_features = pipeline["categorical_features"]
    numerical_features = pipeline["numerical_features"]
    continuous_features = pipeline["continuous_features"]

    feature_order = pipeline["feature_order"]

    drop_columns_corr = pipeline["drop_columns_corr"]
    vif_drop_columns = pipeline["vif_drop_columns"]

    # emp_length_yrs
    data["emp_length_yrs"] = data["emp_length_yrs"].fillna(
        imputation_values["emp_length_yrs"]
    )

    # bureau_score_2
    data["bureau_score_2"] = data["bureau_score_2"].fillna(
        imputation_values["bureau_score_2"]
    )

    # p_days_since_last_pay_cnt
    data["p_days_since_last_pay_cnt"] = data["p_days_since_last_pay_cnt"].fillna(
        imputation_values["p_days_since_last_pay_cnt"]
    )

    # dti_ratio
    data["dti_ratio"] = data["dti_ratio"].fillna(
        imputation_values["dti_ratio"]
    )

    # d_mob_since_last_delinq_cnt
    data.loc[
        (data["d_mob_since_last_delinq_cnt"].isna()) &
        (data["d_times_30dpd_12m_cnt"] == 0),
        "d_mob_since_last_delinq_cnt"
    ] = -1

    data["d_mob_since_last_delinq_cnt"] = (
        data["d_mob_since_last_delinq_cnt"]
        .fillna(imputation_values["d_mob_since_last_delinq_cnt"])
    )

    # collateral_value_amt
    data.loc[
        (data["secured_flag"] == 0) &
        (data["collateral_value_amt"].isna()),
        "collateral_value_amt"
    ] = 0

    data.loc[
        (data["secured_flag"] == 1) &
        (data["collateral_value_amt"].isna()),
        "collateral_value_amt"
    ] = imputation_values["secured_collateral_value"]

    # Drop highly correlated features
    data = data.drop(
    columns=drop_columns_corr,
    errors="ignore",)

    # Drop high VIF features
    data = data.drop(
    columns=vif_drop_columns,
    errors="ignore",)

    # One-hot encode categorical variables
    encoder_features= pipeline["encoder_features"]
    
    encoded = encoder.transform(
        data[encoder_features]
    )
    
    encoded= pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(encoder_features),
        index=data.index,
        )

    # Combine numerical and encoded features
    data = pd.concat(
        [
            data[numerical_features],
            encoded,
            ],
            axis=1,
            )

    # Drop variables identified during model development
    data = data.drop(
        columns=["d_mob_since_last_delinq_cnt"],
        errors="ignore",
        )

    data = data.drop(
        columns=vif_drop_columns,
        errors="ignore",
        )

    data = data.drop(
        columns=[
            "card_type_cd_CONSUMER",
            "card_type_cd_CORPORATE",
            "collateral_type_cd_NONE",
            ],
            errors="ignore",
            )

    # Scale continuous variables
    continuous_to_scale = [
        col for col in continuous_features
        if col in data.columns
        ]

    data[continuous_to_scale] = scaler.transform(
        data[continuous_to_scale]
        )

    # Align feature order used during training
    data = data.reindex(
        columns=feature_order,
        fill_value=0,
        )

    # Add intercept
    import statsmodels.api as sm

    data = sm.add_constant(
        data,
        has_constant="add",
        )

    return data


def build_scorecard_table(
    result_final,
    woe_tables,
    binning_process,
):

    """
    Build a scorecard table from the final PD model.
    """

    import pandas as pd

    # Model coefficients
    coefficients= pd.DataFrame({
        "feature": result_final.params.index,
        "coefficient": result_final.params.values,
    })

    # Remove intercept
    coefficients = coefficients[
        coefficients["feature"] != "const"
    ].copy()

    scorecard_rows = []

    for feature in coefficients["feature"]:

        coefficient= coefficients.loc[
            coefficients["feature"]== feature,
            "coefficient",
        ].iloc[0]

        # Categorical variables
        if feature in woe_tables:

            temp= woe_tables[feature].copy()

            temp= temp[
                [
                    "category",
                    "woe",
                    "iv",
                ]
            ]

            temp["feature"]= feature
            temp["coefficient"]= coefficient

        # Continuous variables
        else:

            binning= (
                binning_process.get_binned_variable(feature)
            )

            temp= (
                binning.binning_table.build()
                )

            temp= temp.rename(
                columns= {
                    "Bin": "category",
                    "WoE": "woe",
                    "IV": "iv",
                    }
            )

            temp= temp[
                [
                    "category",
                    "woe",
                    "iv",
                ]
            ]

            # Remove Totals row
            temp= temp[
                temp["category"].notna()
                ]

            temp= temp[
                temp["category"] != ""
                ]

            # Ensure WoE is numeric
            temp["woe"]= pd.to_numeric(
                temp["woe"],
                errors= "coerce",
                )

            temp= temp.dropna(
                subset=["woe"]
                )

        temp["feature"]= feature
        temp["coefficient"]= coefficient

        scorecard_rows.append(temp)
    

    scorecard= pd.concat(
        scorecard_rows,
        ignore_index= True,
    )

    scorecard= scorecard[
        [
            "feature",
            "category",
            "woe",
            "coefficient",
            "iv"
        ]
    ]

    return scorecard


