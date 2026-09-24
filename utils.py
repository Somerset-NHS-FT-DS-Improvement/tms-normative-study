import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf


def bootstrap_quantile_conf_int(data, q=0.975, n_boot=100, random_state=42):
    # Bootstrap confidence interval for quantiles
    rng = np.random.default_rng(random_state)

    boots = []
    n = len(data)

    for _ in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        boots.append(np.quantile(sample, q))

    ci_lower = np.quantile(boots, 0.025)
    ci_upper = np.quantile(boots, 0.975)

    return ci_lower, ci_upper, n


def fit_and_predict(data, formula, boot_number, prediction_grid, muscle_col):
    # Fit OLS and fetch muscle-specific estimates for upper limit.

    fitted_model = smf.ols(formula, data=data).fit()

    # Calculate residuals
    residual_df = data[[muscle_col]].copy()
    residual_df["observed"] = fitted_model.model.endog
    residual_df["fitted"] = fitted_model.predict(data)

    residual_df["residual"] = residual_df["observed"] - residual_df["fitted"]

    # Fetch 97.5th quantile residual for each muscle
    residual_quantiles = (
        residual_df.groupby(muscle_col, as_index=False)["residual"]
        .quantile(0.975)
        .rename(columns={"residual": "resid_q975"})
    )

    predictions = prediction_grid.copy()

    predictions["prediction"] = fitted_model.predict(prediction_grid)

    predictions = predictions.merge(
        residual_quantiles,
        on=muscle_col,
        how="left",
        validate="many_to_one",
    )

    predictions["upper_limit"] = predictions["prediction"] + predictions["resid_q975"]

    predictions["height"] = [155, 160, 165, 170, 175, 180, 185] * 6

    predictions["boot"] = boot_number

    return fitted_model, predictions


def bootstrap_ols_predictions(
    df,
    formula,
    prediction_grid,
    cluster_col="patient_id",
    muscle_col="muscle",
    height_col="height_unadjusted",
    n_boot=1000,
    random_state=42,
):
    # Estimate height-adjusted 97.5th percentile using OLS and fitted residuals.

    rng = np.random.default_rng(random_state)

    initial_model = smf.ols(formula, data=df).fit()

    analysis_df = (
        df.loc[initial_model.model.data.row_labels]
        .dropna(subset=[cluster_col, muscle_col])
        .copy()
    )

    # Store each patients complete set of observations
    patient_groups = {
        patient_id: patient_df
        for patient_id, patient_df in analysis_df.groupby(
            cluster_col,
            sort=False,
        )
    }

    patient_ids = np.array(list(patient_groups))

    # Point estimate for original model
    model, original_predictions = fit_and_predict(
        analysis_df,
        boot_number=-1,
        formula=formula,
        prediction_grid=prediction_grid,
        muscle_col=muscle_col,
    )

    # Cluster bootstrapping
    bootstrap_predictions = []

    for boot in range(n_boot):

        sampled_ids = rng.choice(
            patient_ids,
            size=len(patient_ids),
            replace=True,
        )

        boot_df = pd.concat(
            [
                patient_groups[patient_id].assign(_boot_patient_id=draw_number)
                for draw_number, patient_id in enumerate(sampled_ids)
            ],
            ignore_index=True,
        )

        _, boot_predictions = fit_and_predict(
            boot_df,
            boot_number=boot,
            formula=formula,
            prediction_grid=prediction_grid,
            muscle_col=muscle_col,
        )

        bootstrap_predictions.append(boot_predictions)

    bootstrap_predictions = pd.concat(
        bootstrap_predictions,
        ignore_index=True,
    )

    return model, original_predictions, bootstrap_predictions


def residual_plot(plot_df):
    # Plot spread of residuals from OLS regression over height.

    plt.rcParams.update(
        {
            "axes.labelsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "legend.title_fontsize": 14,
        }
    )

    sns.set_theme(style="ticks")

    g = sns.FacetGrid(
        plot_df, col="muscle", col_wrap=3, height=4, aspect=1.2, sharey=True
    )

    g.map_dataframe(
        sns.scatterplot,
        x="height_unadjusted",
        y="resid",
        color=sns.color_palette("colorblind")[0],
        alpha=0.5,
        s=40,
    )

    for muscle, ax in g.axes_dict.items():

        ax.axhline(0, color="red", ls="--", lw=1.5)

        ax.axvline(155, color="black", ls=":", alpha=0.8)
        ax.axvline(185, color="black", ls=":", alpha=0.8)

        ax.axvspan(155, 185, color="grey", alpha=0.1)

        # match violin plot styling
        ax.grid(True, which="both", axis="y", alpha=0.3)
        ax.grid(False, axis="x")

        ax.tick_params(axis="both", labelsize=14)

        ax.set_title(muscle, fontsize=14, fontweight="normal")

    g.set_axis_labels("Height (cm)", "Residual CMCT (msec)\nobserved − predicted")

    g.figure.tight_layout(rect=[0, 0.05, 1, 1])

    sns.despine()
    g.figure.tight_layout()

    return g
