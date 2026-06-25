import pandas as pd
import numpy as np
import re
from itertools import combinations
from sklearn.metrics import roc_auc_score,roc_curve, precision_recall_curve, average_precision_score
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy.stats import bootstrap, norm, ttest_rel
from statsmodels.stats.contingency_tables import mcnemar



def _safe_exp(x):
    # Logistic-regression CI bounds on tiny subgroups can be huge enough that
    # np.exp overflows float64. Clip to a value just below the overflow point
    # and return np.inf for anything larger so downstream summaries stay finite.
    arr = np.asarray(x, dtype=float)
    out = np.where(arr > 700, np.inf, np.exp(np.clip(arr, -700, 700)))
    return out.item() if np.ndim(x) == 0 else out

def calculate_mean_ci(
    values,
    confidence=0.95,
    random_state=614,
    n_resample=1000,
    lower=-1000000,
    upper=1000000,
):
    mean = np.mean(values)
    ci_bootstrap = bootstrap(
        data=(values,),
        statistic=np.mean,
        confidence_level=confidence,
        random_state=random_state,
        method='percentile',
        n_resamples=n_resample)
    ci_lower = ci_bootstrap.confidence_interval.low
    ci_upper = ci_bootstrap.confidence_interval.high
    return mean, max(ci_lower, lower), min(ci_upper, upper)

def calculate_confidence_interval(
    values,
    confidence=0.95,
    lower=0,
    upper=1
):
    n = len(values)
    mean = values.mean()
    std = values.std()
    se = std/np.sqrt(n)
    z = norm.ppf(1 - (1 - confidence)/2)
    ci_lower = mean - z*se
    ci_upper = mean + z*se
    return mean, max(ci_lower, lower), min(ci_upper, upper)

def calculate_auc_ci(
    y_true,
    y_score,
    metric_fn=roc_auc_score,
    n_resamples=1000,
    confidence_level=0.95,
    seed=614,
):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score must have the same length")

    # Point estimate
    point = float(metric_fn(y_true, y_score))

    # Statistic function for scipy bootstrap
    def stat(yt, ys):
        if np.unique(yt).size < 2:
            return np.nan  # undefined AUC
        return metric_fn(yt, ys)

    res = bootstrap(
        data=(y_true, y_score),
        statistic=stat,
        paired=True,                 # critical
        vectorized=False,            # our stat handles 1 sample at a time
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method="percentile",
        random_state=seed,
    )
    low = float(res.confidence_interval.low)
    high = float(res.confidence_interval.high)
    return point, (low, high), res

def make_threshold_dataframe(y_true, y_prob):
    ids = y_prob.index
    d_roc = pd.DataFrame(
        np.array(roc_curve(y_true=y_true, y_score=y_prob)).transpose(),
        columns=['False Positive Rate', 'True Positive Rate', 'Threshold']
    ).set_index('Threshold')

    pre, rec, thr = precision_recall_curve(y_true=y_true, y_score=y_prob)
    pre = pre[:-1]
    rec = rec[:-1]
    d_pr = pd.DataFrame(
        np.array([pre, rec, thr]).transpose(),
        columns=['Precision', 'Recall', 'Threshold']
    ).set_index('Threshold')

    dth = pd.concat([d_pr, d_roc], axis=1).iloc[:-1]
    dth['False Positive Rate'] = dth['False Positive Rate'].bfill()
    dth['True Positive Rate'] = dth['True Positive Rate'].bfill()

    dth['Prediction Count'] = [(y_prob >= thr).sum() for thr in dth.index]
    dth['Prediction Rate'] = dth['Prediction Count'] / len(ids)
    return dth

opt_to_color = dict(
    zip(
        [f"Max {metric}" for metric in ['F1', 'F0.5', 'F2', 'Youden J']],
        ['Green', 'Red', 'Blue', 'Orange']
    )
)

def make_opt_to_threshold(threshold_df):
    dth = threshold_df.copy()
    dth.loc[dth.index[(dth['Precision'] * dth['Recall'] / (dth['Precision'] + dth['Recall'])).argmax()], 'Max F1'] = True
    dth.loc[dth.index[(dth['Precision'] * dth['Recall'] / (.25 * dth['Precision'] + dth['Recall'])).argmax()], 'Max F0.5'] = True
    dth.loc[dth.index[(dth['Precision'] * dth['Recall'] / (4 * dth['Precision'] + dth['Recall'])).argmax()], 'Max F2'] = True
    dth.loc[dth.index[(dth['True Positive Rate'] - dth['False Positive Rate']).argmax()], 'Max Youden J'] = True

    opt_to_thr = {col: float(dth[dth[col] == True].index[0]) for col in dth if 'Max' in col}
    return opt_to_thr

def get_closest_threshold_in_array(target_thr, threshold_array):
    return float(threshold_array[
        np.argmin(np.abs(np.array(threshold_array) - target_thr))
        ])

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def make_classifier_curves(y_true, y_prob, opt_to_threshold=None):
    dth = make_threshold_dataframe(y_true=y_true, y_prob=y_prob)
    if opt_to_threshold is None:
        opt_to_threshold = make_opt_to_threshold(dth)
    else:
        pass

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['<b>ROC Curve', '<b>PR Curve', '<b>Recall vs. Prediction Rate', '<b>Precision vs. Prediction Rate'],
        vertical_spacing=.2, horizontal_spacing=.2)
    for xname, yname, row_ix, col_ix in zip(
        ['False Positive Rate', 'Recall', 'Prediction Rate', 'Prediction Rate'],
        ['Recall', 'Precision', 'Recall', 'Precision'],
        [1, 1, 2, 2], [1, 2, 1, 2]
    ):
        fig.add_trace(
            go.Scatter(
                x=dth[xname].to_list(), y=dth[yname].to_list(), text=dth.index.to_list(),
                mode='lines', showlegend=False, marker_color='black'),
            row=row_ix, col=col_ix)

        for opt, thr in opt_to_threshold.items():
            ix = np.argmin(np.abs(dth.index.values - thr))
            fig.add_trace(
                go.Scatter(
                    x=[dth.iloc[ix][xname]], y=[dth.iloc[ix][yname]], text=thr,
                    name=opt,
                    marker_color=opt_to_color.get(opt, 'black'), mode='markers', marker_size=10,
                    showlegend=(row_ix*col_ix == 1)),
                row=row_ix, col=col_ix)

    roc_auc = roc_auc_score(y_score=y_prob, y_true=y_true)
    pr_auc = average_precision_score(y_score=y_prob, y_true=y_true)
    fig.add_annotation(x=.5, y=.15, showarrow=False, text=f"AUC={round(roc_auc, 3)}", col=1, row=1)
    fig.add_annotation(x=.5, y=.15, showarrow=False, text=f"AUC={round(pr_auc, 3)}", col=2, row=1)

    fig.update_xaxes(linecolor='black', ticks='outside', range=[0, 1.01], dtick=.25)
    fig.update_yaxes(linecolor='black', ticks='outside', range=[0, 1.01], dtick=.25, gridcolor='lightgray')

    fig.update_yaxes(title_text='Recall', col=1)
    fig.update_yaxes(title_text='Precision', col=2)
    fig.update_xaxes(title_text='False Positive Rate', row=1, col=1)
    fig.update_xaxes(title_text='Recall', row=1, col=2)
    fig.update_xaxes(title_text='Prediction Rate', row=2)

    fig.update_layout(
        height=900, width=1000, plot_bgcolor='white',
        font_size=18, font_family='Arial', font_color='black')
    fig.add_annotation(x=-.12, y=1.08, xref='paper', yref='paper', text='(a)', showarrow=False)
    fig.add_annotation(x=.48, y=1.08, xref='paper', yref='paper', text='(b)', showarrow=False)
    fig.add_annotation(x=-.12, y=.46, xref='paper', yref='paper', text='(c)', showarrow=False)
    fig.add_annotation(x=.48, y=.46, xref='paper', yref='paper', text='(d)', showarrow=False)
    fig.update_annotations(font_size=24)
    fig.update_annotations(selector=dict(y=.15), font_size=18)

    return fig


def decision_curve(y_true, y_prob, thresholds=None):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    N = len(y_true)
    prevalence = y_true.mean()
    nb_model = []
    nb_all = []
    nb_none = np.zeros_like(thresholds)

    for pt in thresholds:
        preds = (y_prob >= pt)

        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))

        weight = pt / (1 - pt)

        nb_model.append((tp/N) - weight*(fp/N))

        # Treat-all strategy
        nb_all.append(prevalence - weight*(1 - prevalence))

    return thresholds, np.array(nb_model), np.array(nb_all), nb_none

def make_horiz_error_bar_fig(
    data:pd.DataFrame,
    attr:str,
    val_key:str,
    group_order:list,
    confidence=.95
):
    yy = group_order
    xx = []
    err_over = []
    err_under = []
    for grp in group_order:
        x, e_under, e_over = calculate_mean_ci(
            data[data[attr] == grp][val_key], 
            confidence=confidence,
            lower=0,
            upper=1,
        )
        xx.append(x)
        err_under.append(x-e_under)
        err_over.append(e_over-x)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        orientation='h',
        y=yy, x=xx,
        error_x=dict(symmetric=False, array=err_over, arrayminus=err_under, color='darkblue'),
        marker_color='lightgray', marker_line_color='darkblue',
    ))
    fig.update_yaxes(tickvals=group_order, ticktext=[f"{grp} ({data[attr].value_counts().get(grp, 0)})" for grp in group_order])
    return fig

import pandas as pd
import plotly.graph_objects as go

def make_ci_error_bar_plot(
    data: pd.DataFrame,
    attr: str,
    val_key: str,
    group_order: list = None,
    color='darkblue',
    mode='mean',
    lower=1,
    upper=1000,
):

    # subgroup counts
    counts = data[attr].value_counts()

    # determine plotting order
    if group_order is None:
        group_order = counts.index.tolist()

    # sort by descending subgroup size
    group_order = sorted(
        group_order,
        key=lambda g: counts.get(g, 0),
        reverse=True
    )

    yy = []
    xx = []
    err_over = []
    err_under = []

    for grp in group_order:

        data_grp = data[data[attr] == grp]

        if len(data_grp) == 0:
            continue

        if mode == 'mean':
            x, e_under, e_over = calculate_mean_ci(
                data_grp[val_key],
                lower=lower,
                upper=upper
            )

        elif mode == 'median':
            x = data_grp[val_key].median()
            e_under = data_grp[val_key].quantile(.25)
            e_over = data_grp[val_key].quantile(.75)

        xx.append(x)

        # distances from center point
        err_under.append(max(x - e_under, 1))
        err_over.append(e_over - x)

        # label with counts
        yy.append(f"{grp} ({counts.get(grp, 0)})")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            mode='markers',
            y=yy,
            x=xx,
            error_x=dict(
                symmetric=False,
                array=err_over,
                arrayminus=err_under,
                color=color
            ),
            marker_color=color,
            marker_line_color=color,
        )
    )

    # largest group at top
    #fig.update_yaxes(autorange="reversed")

    return fig

def run_bias_models(
    df,
    attributes,
    target_key="target_var",
    pred_keys=None,
    reference_map=None,
    correction_method="fdr_bh",
    alpha=0.05,
    correction_scope="within_attribute_metric_model",
    metrics=['prevalence', 'positive_rate', 'recall', 'precision']
):
    """
    Run univariate logistic regression bias models across subgroup attributes.

    Metrics included:
      - prevalence: target_key ~ C(attribute)
      - positive_rate: pred_key ~ C(attribute) for each pred_key
      - recall: pred_key ~ C(attribute), restricted to target_key == 1
      - precision: target_key ~ C(attribute), restricted to pred_key == 1

    Parameters
    ----------
    df : pd.DataFrame
    attributes : list[str]
        Demographic/grouping variables, e.g. ["sex", "race", "ethnicity", "gestational_age"]
    target_key : str
        Binary true label column.
    pred_keys : list[str]
        Binary prediction columns for models.
    reference_map : dict[str, object] or None
        Reference group for each attribute, e.g. {"sex": "Male", "race": "White"}
    correction_method : str
        Method for statsmodels.stats.multitest.multipletests
    alpha : float
    correction_scope : str
        One of:
        - "global": correct over all rows
        - "within_attribute": correct within each attribute
        - "within_attribute_metric": correct within each attribute x metric
        - "within_attribute_metric_model": correct within each attribute x metric x model_label

    Returns
    -------
    pd.DataFrame
    """

    if pred_keys is None:
        pred_keys = []

    assert isinstance(target_key, str), f"target_key must be str, got {type(target_key)}"
    assert all(isinstance(a, str) for a in attributes), "All attributes must be strings"
    assert all(isinstance(p, str) for p in pred_keys), "All pred_keys must be strings"

    results = []

    def _set_reference(data, attr):
        if reference_map and attr in reference_map:
            ref = reference_map[attr]
            observed = list(pd.Series(data[attr].dropna().unique()))
            if ref in observed:
                cats = [ref] + [x for x in observed if x != ref]
                data[attr] = pd.Categorical(data[attr], categories=cats, ordered=True)
        return data

    def _extract_level(term, attr):
        m = re.match(rf"C\({attr}\)\[T\.([^\]]+)\]", term)
        return m.group(1) if m else term

    def _fit_logit(data, formula):
        return smf.logit(formula, data=data).fit(disp=False)

    def _append_results(summary, attr, metric, outcome_col, subset_label, model_label=None):
        for term, row in summary.iterrows():
            if term == "Intercept":
                continue
            coef = row["Coef."]
            results.append({
                "attribute": attr,
                "metric": metric,
                "model_label": model_label,
                "outcome_col": outcome_col,
                "subset": subset_label,
                "term": term,
                "level": _extract_level(term, attr),
                "OR": _safe_exp(coef),
                "CI_lower": _safe_exp(row["[0.025"]),
                "CI_upper": _safe_exp(row["0.975]"]),
                "p_value": row["P>|z|"],
                "significant": False,
                "p_adj": np.nan,
            })

    for attr in attributes:
        # 1) prevalence
        data = df[[target_key, attr]].dropna().copy()
        if 'prevalence' in metrics and not data.empty:
            data[target_key] = data[target_key].astype(int)
            data = _set_reference(data, attr)
            formula = f"{target_key} ~ C({attr})"
            try:
                model = _fit_logit(data, formula)
                _append_results(
                    model.summary2().tables[1],
                    attr=attr,
                    metric="prevalence",
                    outcome_col=target_key,
                    subset_label="full_population",
                    model_label='n/a',
                )
            except Exception as e:
                print(f"Skipping prevalence for {attr}: {e}")

        # 2) positive_rate for each model
        for pred_key in pred_keys:
            if 'positive_rate' not in metrics:
                break
            data = df[[pred_key, attr]].dropna().copy()
            if data.empty:
                continue
            data[pred_key] = data[pred_key].astype(int)
            data = _set_reference(data, attr)
            formula = f"{pred_key} ~ C({attr})"
            try:
                model = _fit_logit(data, formula)
                _append_results(
                    model.summary2().tables[1],
                    attr=attr,
                    metric="positive_rate",
                    outcome_col=pred_key,
                    subset_label="full_population",
                    model_label=pred_key,
                )
            except Exception as e:
                print(f"Skipping positive_rate for {attr}, {pred_key}: {e}")

        # 3) recall for each model
        for pred_key in pred_keys:
            if 'recall' not in metrics:
                break
            data = df[[target_key, pred_key, attr]].dropna().copy()
            data = data.query(f"{target_key} == 1").copy()
            if data.empty:
                continue
            data[pred_key] = data[pred_key].astype(int)
            data = _set_reference(data, attr)
            formula = f"{pred_key} ~ C({attr})"
            try:
                model = _fit_logit(data, formula)
                _append_results(
                    model.summary2().tables[1],
                    attr=attr,
                    metric="recall",
                    outcome_col=pred_key,
                    subset_label=f"{target_key} == 1",
                    model_label=pred_key,
                )
            except Exception as e:
                print(f"Skipping recall for {attr}, {pred_key}: {e}")

        # 4) precision for each model
        for pred_key in pred_keys:
            if 'precision' not in metrics:
                break
            data = df[[target_key, pred_key, attr]].dropna().copy()
            data = data.query(f"{pred_key} == 1").copy()
            if data.empty:
                continue
            data[target_key] = data[target_key].astype(int)
            data = _set_reference(data, attr)
            formula = f"{target_key} ~ C({attr})"
            try:
                model = _fit_logit(data, formula)
                _append_results(
                    model.summary2().tables[1],
                    attr=attr,
                    metric="precision",
                    outcome_col=target_key,
                    subset_label=f"{pred_key} == 1",
                    model_label=pred_key,
                )
            except Exception as e:
                print(f"Skipping precision for {attr}, {pred_key}: {e}")

    results_df = pd.DataFrame(results)

    if results_df.empty:
        return results_df

    if correction_scope == "global":
        group_cols = None
    elif correction_scope == "within_attribute":
        group_cols = ["attribute"]
    elif correction_scope == "within_attribute_metric":
        group_cols = ["attribute", "metric"]
    elif correction_scope == "within_attribute_metric_model":
        group_cols = ["attribute", "metric", "model_label"]
    elif correction_scope == "within_metric_model":
        group_cols = ["metric", "model_label"]
    else:
        raise ValueError(f"Unknown correction_scope: {correction_scope}")

    if group_cols is None:
        reject, p_adj, _, _ = multipletests(
            results_df["p_value"].values,
            alpha=alpha,
            method=correction_method,
        )
        results_df["p_adj"] = p_adj
        results_df["significant"] = reject
    else:
        for _, idx in results_df.groupby(group_cols).groups.items():
            pvals = results_df.loc[idx, "p_value"].values
            reject, p_adj, _, _ = multipletests(
                pvals,
                alpha=alpha,
                method=correction_method,
            )
            results_df.loc[idx, "p_adj"] = p_adj
            results_df.loc[idx, "significant"] = reject

    return results_df.sort_values(
        ["attribute", "metric", "model_label", "p_adj", "p_value"],
        na_position="last",
    ).reset_index(drop=True)