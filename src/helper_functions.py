import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_woe_iv(data, feature, target):

    woe_table = (
        data.groupby(feature)[target].agg(
            total_accounts='count',
            bad_accounts='sum'
        ).reset_index()
    )

    woe_table['good_accounts'] = (
        woe_table['total_accounts'] - woe_table['bad_accounts']
    )

    woe_table['good_pct'] = (
        (woe_table['good_accounts'] + 0.5) /
        (woe_table['good_accounts'].sum() + 0.5 * len(woe_table))
    )

    woe_table['bad_pct'] = (
        (woe_table['bad_accounts'] + 0.5) /
        (woe_table['bad_accounts'].sum() + 0.5 * len(woe_table))
    )

    woe_table['woe'] = np.log(
        woe_table['good_pct'] / woe_table['bad_pct']
    )

    woe_table['iv'] = (
        woe_table['good_pct'] - woe_table['bad_pct']
    ) * woe_table['woe']

    iv = woe_table['iv'].sum()

    return woe_table, iv

#=======================================================================================#

def transform_categorical_to_woe(data, features, woe_tables):
    data_woe= pd.DataFrame(index= data.index)

    for feature in features:
        woe_mapping= (
            woe_tables[feature].set_index('category')['woe']
        )

        data_woe[feature]= data[feature].map(woe_mapping).fillna(0)

    return data_woe

#=========================================================================================#

# decile table

def create_decile_table(y_true, y_pred_proba):
    
    # Create DataFrame
    df = pd.DataFrame({
        'actual': y_true,
        'predicted_pd': y_pred_proba
    })

    # Sort by predicted PD (highest risk first)
    df = df.sort_values('predicted_pd', ascending=False).reset_index(drop=True)

    # Create deciles
    df['decile'] = pd.qcut(df.index, 10, labels=range(1, 11))

    # Aggregate statistics
    decile_table = (
        df.groupby('decile', observed=False)
        .agg(
            total_accounts=('actual', 'count'),
            total_bads=('actual', 'sum'),
            avg_predicted_pd=('predicted_pd', 'mean')
        )
        .reset_index()
    )

    # Calculate additional metrics
    decile_table['total_goods'] = (
        decile_table['total_accounts'] - decile_table['total_bads']
    )

    decile_table['observed_bad_rate'] = (
        decile_table['total_bads'] / decile_table['total_accounts']
    )

    decile_table['cum_bads'] = decile_table['total_bads'].cumsum()
    decile_table['cum_goods'] = decile_table['total_goods'].cumsum()

    total_bads = decile_table['total_bads'].sum()
    total_goods = decile_table['total_goods'].sum()

    decile_table['cum_bad_pct'] = (
        decile_table['cum_bads'] / total_bads
    )

    decile_table['cum_good_pct'] = (
        decile_table['cum_goods'] / total_goods
    )

    decile_table['ks'] = (
        decile_table['cum_bad_pct'] - decile_table['cum_good_pct']
    ).abs()

    overall_bad_rate = total_bads / decile_table['total_accounts'].sum()

    decile_table['lift'] = (
        decile_table['observed_bad_rate'] / overall_bad_rate
    )

    return decile_table

#============================================================================================#

# ==============================
# Calibration Check
# ==============================

def plot_calibration(decile_table, dataset_name):
    
    plt.figure(figsize=(6,4))

    plt.plot(
        decile_table['decile'],
        decile_table['avg_predicted_pd'],
        marker='o',
        label='Average Predicted PD'
    )

    plt.plot(
        decile_table['decile'],
        decile_table['observed_bad_rate'],
        marker='s',
        label='Observed Bad Rate'
    )

    plt.xlabel('Decile (1 = Highest Risk)')
    plt.ylabel('Probability of Default')
    plt.title(f'Calibration Check - {dataset_name}')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.show()