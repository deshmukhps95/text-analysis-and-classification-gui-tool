import pandas as pd
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt


def apply_cross_validation(
    features,
    labels,
    k_folds,
    use_svm,
    use_naive_bayes,
    use_random_forest,
    use_logistic_regression,
    use_xgboost,
    use_gradient_boosting,
    plot_cv_graph,
):
    model_list = list()
    if use_svm:
        from classification_models.support_vector_machines import linear_svm

        model_list.append(linear_svm)

    if use_naive_bayes:
        from classification_models.naive_bayes import naive_bayes

        model_list.append(naive_bayes)

    if use_random_forest:
        from classification_models.random_forests import random_forest

        model_list.append(random_forest)

    if use_logistic_regression:
        from classification_models.logistic_regression import logistic_regression

        model_list.append(logistic_regression)

    if use_xgboost:
        from classification_models.xgboosting import xgboost

        model_list.append(xgboost)

    if use_gradient_boosting:
        from classification_models.gradient_boosting import gradient_boost

        model_list.append(gradient_boost)

    entries = []
    for model in model_list:
        model_name = model.__class__.__name__
        accuracies = cross_val_score(
            model, features, labels, scoring="accuracy", cv=k_folds
        )
        for fold_idx, accuracy in enumerate(accuracies):
            print(model_name, ":", accuracy, " ", fold_idx)
            entries.append((model_name, fold_idx, accuracy))
    if plot_cv_graph:
        cv_df = pd.DataFrame(entries, columns=["model_name", "fold_idx", "accuracy"])
        sns.boxplot(x="model_name", y="accuracy", data=cv_df)
        sns.stripplot(
            x="model_name",
            y="accuracy",
            data=cv_df,
            size=8,
            jitter=True,
            edgecolor="red",
            linewidth=2,
        )
        plt.show()
        print(cv_df.groupby("model_name").accuracy.mean())
