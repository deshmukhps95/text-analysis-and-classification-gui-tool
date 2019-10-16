import time
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from preprocessing import TextPreProcessor
from classification_models.support_vector_machines import linear_svm
from classification_models.logistic_regression import logistic_regression
from classification_models.naive_bayes import naive_bayes
from classification_models.random_forests import random_forest
from classification_models.xgboosting import xgboost
from classification_models.gradient_boosting import gradient_boost


class Trainer:
    def __init__(
        self,
        train_file_path,
        val_file_path,
        stop_words_file_path,
        model_name,
        feature_extractor,
    ):
        self.train_file_path = train_file_path
        self.val_file_path = val_file_path
        self.text_preprocessor = TextPreProcessor(
            stop_words_file_path=stop_words_file_path
        )
        self.feature_extractor = feature_extractor
        self.training_model = self.load_model(model_name)

    @staticmethod
    def load_model(model_name):
        model_dict = {
            "Linear SVM": linear_svm,
            "Naive Bayes": naive_bayes,
            "Logistic Regression": logistic_regression,
            "Random Forest": random_forest,
            "XGBoosting": xgboost,
            "Gradient Boosting": gradient_boost,
        }
        return model_dict[model_name]

    @staticmethod
    def plot_confusion_matrix(y_test, y_pred, class_id_df):
        conf_mat = confusion_matrix(y_test, y_pred)
        plt.subplots(figsize=(10, 10))
        sns.heatmap(
            conf_mat,
            annot=True,
            fmt="d",
            xticklabels=class_id_df.Type.values,
            yticklabels=class_id_df.Type.values,
        )
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.show()

    def train(self, training_data, split_test_size, vectorizer_name, get_classification_report, get_confusion_matrix):
        training_data["sentence"] = training_data["sentence"].map(
            self.text_preprocessor.process
        )
        x_train = training_data["sentence"].values
        y_train = training_data["class"].values

        training_features = self.feature_extractor.get_features_for_training(
            x_train, vectorizer_name
        )
        training_labels = y_train
        x_train, x_test, y_train, y_test = train_test_split(
            training_features,
            training_labels,
            test_size=split_test_size,
            random_state=0,
        )
        print(f"Size of test sentences: {training_features.shape}")

        print("Training started")

        t = time.time()
        self.training_model.fit(training_features, training_labels)
        print(f"Training time: {(time.time()-t)/60.0} minutes")

        prediction_labels = self.training_model.predict(x_test)
        print(f"Test Accuracy: {metrics.accuracy_score(y_test, prediction_labels)}")
        if get_classification_report:
            print(
                metrics.classification_report(
                    y_test, prediction_labels, target_names=training_data["Type"].unique()
                )
            )
        if get_confusion_matrix:
            class_id_df = training_data[["class", "class_id"]].drop_duplicates().sort_values("class_id")
            self.plot_confusion_matrix(y_test, prediction_labels, class_id_df)

    def validate(self, validation_data, vectorizer_name):
        # pre-process the validation data
        validation_data["sentence"] = validation_data["sentence"].map(
            self.text_preprocessor.process
        )

        x_val = validation_data["sentence"].values
        y_val = validation_data["class"].values

        validation_features = self.feature_extractor.get_features_for_testing(
            x_val, vectorizer_name
        )
        validation_labels = y_val

        val_prediction = self.training_model.predict(validation_features)
        print(
            f"Validation Accuracy: {metrics.accuracy_score(validation_labels, val_prediction)}"
        )

    def save_trained_model(self, model_checkpoint_path):
        file = open(model_checkpoint_path, "wb")
        pickle.dump(self.training_model, file, protocol=3)
        pickle.dump(self.feature_extractor, file, protocol=3)
