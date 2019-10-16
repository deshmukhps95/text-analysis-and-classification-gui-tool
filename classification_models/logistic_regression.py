from sklearn.linear_model import LogisticRegression

logistic_regression = LogisticRegression(
    penalty="l1",
    dual=False,
    C=0.40,
    fit_intercept=True,
    class_weight="balanced",
    random_state=42,
    solver="liblinear",
    max_iter=200,
    multi_class="ovr",
    warm_start=False,
    n_jobs=2,
)
