**Credit Risk Modeling: PD, LGD, EAD & IFRS 9 ECL**

An end-to-end credit risk model suite built on a simulated credit card portfolio (1,000,000 account-month records), covering Behavioral PD, LGD, EAD, and a full IFRS 9-aligned Expected Credit Loss calculation from raw data through model development, validation, and portfolio-level reserve estimation.

**Project Structure**
notebooks/
  01_behavioural_pd_model.ipynb   Behavioral PD scorecard (WOE/IV, logistic regression)
  02_LGD_model.ipynb              Loss Given Default (Beta Regression)
  03_EAD_model.ipynb              Exposure at Default / CCF modeling
  04_ECL_model.ipynb              Full-portfolio IFRS 9 ECL calculation

src/
  preprocessing.py                Shared data loading, cleaning, and scoring pipeline
  helper_functions.py             WOE/IV, decile table, and calibration utilities

saved_objects/
  pd_scorecard.csv                Final PD scorecard (points per variable/bin)

Raw and processed data files, and large model pickles, are excluded via .gitignore (regeneratable / too large for version control), see Data below.

**Methodology**

The project follows a standard bank credit-risk model development lifecycle:

**Data preparation**: data quality checks (duplicate detection, business-rule consistency checks), sentinel handling, and population-specific missing value treatment (e.g., conditional imputation for delinquency-history fields, secured-vs-unsecured logic for collateral values), with all statistics fit on the Development sample only and applied to Validation/OOT to prevent leakage.

**Population definitions**

PD: accounts not currently in default (behavioral model predicts which currently-performing accounts default within 12 months)
LGD / EAD: accounts currently in default (realized loss severity and exposure can only be observed after default occurs)
ECL: the full portfolio, performing and defaulted accounts combined, as required under IFRS 9

**Model selection**: logistic regression (WOE-transformed) for PD, since it's the industry-standard interpretable scorecard approach; Beta Regression for LGD and EAD, since both targets are continuous and bounded in [0, 1].

**Notebook 1: Behavioral PD Model**

WOE binning and Information Value analysis (categorical/discrete via manual binning with sparse-bin merging; continuous via optbinning's BinningProcess), correlation and VIF-based multicollinearity reduction, and a logistic regression fit refined via backward elimination on statistical significance (p < 0.05). Concludes with a full points-based scorecard (PDO scaling).

Results (Out-of-Time):

Metric	Train	Validation	OOT
AUC	    0.789	  0.784	    0.787
Gini	0.578	  0.569	    0.574
KS	    0.433	  0.429	    0.435

Stable performance across all three samples indicates good generalization with no meaningful overfitting. Model validated with decile/gains tables, rank-order checks, and calibration analysis.

**Notebook 2: LGD Model**

Beta Regression fit on the defaulted population, with the target constructed as realized loss (1 − recovery_amt / EAD). Initial fitting failed to converge; resolved by standardizing continuous predictors before MLE estimation.

Results (Out-of-Time): MAE ≈ 0.072, R² ≈ 0.31–0.34, Spearman ≈ 0.37–0.41 , consistent across Development/Validation/OOT.

Key drivers: secured_flag and b_util_ratio were the strongest significant predictors. Several state-level variables were also significant, suggesting geographic variation in recovery outcomes worth further investigation. Model diagnostics (actual-vs-predicted, residual analysis) show the model compresses predictions toward the population mean, under-differentiating extreme LGD outcomes.

**Notebook 3: EAD Model**

Models the Credit Conversion Factor (CCF), the proportion of undrawn credit converted into exposure before default using the same Beta Regression approach as LGD.

Result: the model provided no meaningful predictive power (R² ≈ 0, Spearman ≈ 0 across all samples). To test this rigorously, the fitted model was benchmarked against a simple segment-average CCF baseline; the baseline performed equal to or better than the regression model on Validation and OOT.

Decision: segment-average CCF was selected as the production approach, based on the validation evidence rather than model complexity, a deliberate example of model governance (build → benchmark → deploy the justified choice, not the more complex one by default).

**Notebook 4: Expected Credit Loss (IFRS 9)**

Combines all three models to calculate portfolio-level ECL using ECL = PD × LGD × EAD, applied across the entire Out-of-Time portfolio both performing and already-defaulted accounts, as IFRS 9 requires.

Performing accounts: scored using the fitted PD model; LGD and EAD applied as forward-looking assumptions using the same fitted LGD model and segment-average CCF
Already-defaulted accounts: PD = 1 (default has already occurred), LGD from the same fitted model applied to current account characteristics, EAD from actual realized exposure

**Portfolio-level results:**

	            Performing	     Defaulted	      Total
Accounts         243,959	      5,506	          249,465
Avg PD	         3.58%	          100%	          5.71%
Avg LGD	         86.92%	          79.75%	      86.76%
Avg EAD	         4,850.26	      5,740.66        4,869.91
Portfolio ECL	 $38,252,008.61	  $25,180,773.22  $63,432,781.83

Notable finding: although already-defaulted accounts represent only 2.2% of the portfolio by count, they contribute approximately 39.7% of total portfolio ECL, a concrete demonstration of why IFRS 9 mandates including Stage 3 accounts in the reserve calculation, rather than treating ECL as a performing-portfolio-only exercise.

**Data**

This project uses a fully synthetic credit card portfolio dataset (1,000,000 account-month records, 88 raw columns) designed to mirror the structure and naming conventions of a real bank credit risk data mart. Raw and processed data files are excluded from this repository via .gitignore due to size.

Setup
bash
pip install -r requirements.txt

Run the notebooks in order (01 → 04); each notebook saves reusable artifacts (saved_objects/) consumed by the next.

**Tech Stack**

pandas · numpy · scikit-learn · statsmodels · optbinning · matplotlib · seaborn · scipy
