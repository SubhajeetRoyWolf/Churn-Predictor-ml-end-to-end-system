# Model Evaluation Report

## Headline Results (full Olist dataset, ~115K rows)

| Configuration                       | ROC-AUC |
| ----------------------------------- | ------- |
| RF baseline (n=100, max_depth=10)   | 0.72    |
| RF with n=200, class_weight=balanced | 0.79   |
| RF tuned via RandomizedSearchCV     | 0.77 (test) / 0.86 (CV) |

Best tuned params (from RandomizedSearchCV):
- n_estimators: 200
- max_depth: 20
- min_samples_split: 5
- min_samples_leaf: 2
- class_weight: balanced

## Features (7)

Numeric raw: `price`, `freight_value`, `delivery_time`
Engineered: `total_cost = price + freight_value`, `freight_ratio = freight_value / (price + 1)`,
`is_expensive = price > median`, `is_delayed = is_late OR delivery > median`

## Top Features (by SHAP)

Visible in `artifacts/shap_feature_importance.png`. The strongest predictors are
delivery-related signals (`is_delayed`, `delivery_time`) followed by freight ratio and total cost.

## Honest Caveats

- Target is engineered (`review_score <= 2`) as proxy for churn; not true churn.
- Class imbalance ~15% — `class_weight='balanced'` and probability-based thresholding mitigate this.
- A 0.72-0.77 ROC-AUC is solid for a tabular baseline; review-text NLP features would likely push this past 0.80.
