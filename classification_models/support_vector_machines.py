from sklearn.svm import LinearSVC, SVC

svm = SVC(
    kernel="poly",
    degree=2,
    gamma="scale",
    probability=True,
    class_weight="balanced",
)

linear_svm = LinearSVC(
    loss="hinge",
    class_weight="balanced",
    multi_class="crammer_singer",
    max_iter=1200,
)
