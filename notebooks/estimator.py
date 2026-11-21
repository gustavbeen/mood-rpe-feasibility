"""Shared fitting function from notebook 6."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t


def fit_intervals(data, predictor_names):
    """Fit the within-play model and return estimates and uncertainty.

    Each row is a rating interval. Mood components and interaction products
    have already been calculated. Each play represents one participant.
    """
    # stop if any required values are missing; don't silently drop rows
    required = ["play_id", "interval", "next_happiness"] + predictor_names
    assert not data[required].isna().any().any(), "Required values are missing."

    # one indicator for each interval position, except the first
    positions = pd.get_dummies(
        data["interval"],
        prefix="position",
        drop_first=True,
        dtype=float,
    )

    # combine the main predictors and interval-position indicators
    X = pd.concat([data[predictor_names], positions], axis=1)
    groups = data["play_id"]

    # subtract play means from every predictor and the outcome
    # interaction products are already formed before this step
    X_within = X - X.groupby(groups).transform("mean")
    y_within = (
        data["next_happiness"]
        - data.groupby("play_id")["next_happiness"].transform("mean")
    )

    n = len(data)
    n_plays = groups.nunique()
    rank = np.linalg.matrix_rank(X_within.to_numpy())
    assert rank == X_within.shape[1], "Design is rank-deficient after removing participant means."
    assert n_plays > 1  # need more than one play for clustered uncertainty
    assert n > rank

    # fit with uncertainty clustered by play
    fit = sm.OLS(y_within, X_within).fit(
        cov_type="cluster",
        cov_kwds={"groups": groups, "use_correction": False},
    )

    # apply the finite-sample correction, excluding play intercepts
    correction = (
        n_plays / (n_plays - 1)
        * (n - 1) / (n - rank)
    )
    covariance = fit.cov_params() * correction

    # diagonal entries are coefficient variances; square roots give SEs
    se = pd.Series(
        np.sqrt(np.diag(covariance)),
        index=fit.params.index,
    )

    # multiplier for two-sided 95% confidence intervals
    multiplier = t.ppf(0.975, df=n_plays - 1)

    summary = pd.DataFrame({
        "estimate": fit.params,
        "SE": se,
        "CI_low": fit.params - multiplier * se,
        "CI_high": fit.params + multiplier * se,
    })

    # retain all terms, including position effects, with their column names
    return {
        "summary": summary,
        "covariance": covariance,
    }
