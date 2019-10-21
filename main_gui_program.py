from os import path
from gooey import GooeyParser, Gooey
from data_loader import load_training_data, load_validation_data
from data_analysis import DataAnalyser
from preprocessing import TextPreProcessor
from feature_extraction import FeatureExtractor
from training import Trainer
from prediction import Predictor
from model_selection import apply_cross_validation


@Gooey(
    program_name="Text Analysis and Classification Tool",
    default_size=(1150, 650),
    optional_cols=2,
    navigation="SIDEBAR",
    sidebar_title="Classification Pipeline",
    body_bg_color="#f0eeef",
    header_bg_color="#f0eeef",
    footer_bg_color="#f0eeef",
    sidebar_bg_color="#f0eeef",
    terminal_font_size=10,
    menu=[{"name": "Help", "items": []}],
)
def get_parser():
    parent_parser = GooeyParser(add_help=False)
    """parser for sub-parser """
    parser = GooeyParser(add_help=False)

    sub_parser = parser.add_subparsers()
    "sub-parser for analysing data"
    sub_parser_data_analysis = sub_parser.add_parser(
        "Data-Analysis", parents=[parent_parser], help="args parse for cnn request"
    )
    sub_parser_data_analysis.set_defaults(pipeline_type="analysis")
    sub_parser_data_analysis.add_argument(
        "--input_file_path",
        metavar="File Path",
        type=str,
        help="Please provide file to be analysed.",
        default="",
        required=True,
        widget="FileChooser",
    )
    sub_parser_data_analysis.add_argument(
        "--stopwords_file_path",
        metavar="Stopwords File",
        type=str,
        help="Please provide file for stopwords.",
        default="",
        widget="FileChooser",
    )
    sub_parser_data_analysis.add_argument(
        "--word_thresh",
        metavar="Word Threshold",
        type=int,
        default=100,
        help="Provide number of important words to be visualised.",
    )
    sub_parser_data_analysis.add_argument(
        "--word_cloud",
        metavar="Word Cloud",
        action="store_true",
        help="Generate word cloud for input file.",
    )
    sub_parser_data_analysis.add_argument(
        "--plot_bar",
        metavar="Data Distribution",
        action="store_true",
        help="Generate plot for visualising data distribution.",
    )
    sub_parser_data_cross_validation = sub_parser.add_parser(
        "Model-Selection", parents=[parent_parser], help="args parse for cnn request"
    )
    sub_parser_data_cross_validation.set_defaults(pipeline_type="model_selection")
    sub_parser_data_cross_validation.add_argument(
        "--kfolds",
        metavar="Folds",
        type=int,
        help="Provides number of training-data folds",
        choices=range(2, 21),
        gooey_options={
            "validator": {
                "test": "2 <= int(user_input) <= 20",
                "message": "must be in range 2 and 20",
            }
        },
    )
    sub_parser_data_cross_validation.add_argument(
        "--train_file_path",
        metavar="Training File",
        type=str,
        help="provide training file in csv format",
        default="",
        required=True,
        widget="FileChooser",
    )
    sub_parser_data_cross_validation.add_argument(
        "--vectorizer",
        metavar="Vectorizer",
        type=str,
        help="Select a vectorizer to convert text data into numerical data.",
        default="TF-IDF",
        choices=["Count", "TF-IDF", "Hash"],
        required=True,
    )
    sub_parser_data_cross_validation.add_argument(
        "--stopwords_file_path",
        metavar="Stopwords File",
        type=str,
        help="Please provide file for stopwords.",
        default="",
        required=True,
        widget="FileChooser",
    )
    sub_parser_data_cross_validation.add_argument(
        "--use_svm", metavar="SVM", action="store_true", help="Use linear svc"
    )
    sub_parser_data_cross_validation.add_argument(
        "--use_naive_bayes",
        metavar="Naive Bayes",
        action="store_true",
        help="Use naive bayes classifier.",
    )
    sub_parser_data_cross_validation.add_argument(
        "--use_random_forest",
        metavar="Random Forest",
        action="store_true",
        help="Use random forest classifier.",
    )
    sub_parser_data_cross_validation.add_argument(
        "--use_logistic_regression",
        metavar="Logistic Regression",
        action="store_true",
        help="Use logistic regression classifier.",
    )
    sub_parser_data_cross_validation.add_argument(
        "--use_xgboost",
        metavar="XGBoost",
        action="store_true",
        help="Use XGBoost classifier.",
    )
    sub_parser_data_cross_validation.add_argument(
        "--use_gradient_boosting",
        metavar="Gradient Boost",
        action="store_true",
        help="Use gradient boost classifier.",
    )
    sub_parser_training = sub_parser.add_parser(
        "Training", parents=[parent_parser], help="args parse for cnn request"
    )
    sub_parser_training.set_defaults(pipeline_type="training")
    sub_parser_training.add_argument(
        "--train_file_path",
        metavar="Training File",
        type=str,
        help="Provide training file in csv format.",
        default="",
        required=True,
        widget="FileChooser",
    )
    sub_parser_training.add_argument(
        "--val_file_path",
        metavar="Validation File",
        type=str,
        help="Provide validation file in csv format.",
        default="",
        required=True,
        widget="FileChooser",
    )
    sub_parser_training.add_argument(
        "--best_model",
        metavar="Classification Model",
        type=str,
        help="Select best model after k-fold cross validation.",
        default="Linear SVM",
        choices=[
            "Random Forest",
            "Linear SVM",
            "Naive Bayes",
            "Logistic Regression",
            "XGBoosting",
            "Gradient Boosting",
        ],
        required=True,
    )
    sub_parser_training.add_argument(
        "--vectorizer",
        metavar="Vectorizer",
        type=str,
        help="Select a vectorizer to convert text data into numerical data.",
        default="TF-IDF",
        choices=["Count", "TF-IDF", "Hash"],
        required=True,
    )
    sub_parser_training.add_argument(
        "--split_size",
        metavar="Split Size",
        type=float,
        help="Provide test split size.",
        gooey_options={
            "validator": {
                "test": "0.0 <= float(user_input) <= 0.50",
                "message": "must be in range 0.0 and 0.50",
            }
        },
    )
    sub_parser_training.add_argument(
        "--model_check_point_path",
        metavar="Save Model",
        type=str,
        help="Choose a folder to save the model.",
        default="",
        widget="DirChooser",
    )
    sub_parser_training.add_argument(
        "--get_classification_report",
        metavar="Classification Report",
        action="store_true",
        help="Use this flag to get classification report.",
    )
    sub_parser_training.add_argument(
        "--get_confusion_matrix",
        metavar="Confusion Matrix",
        action="store_true",
        help="Use this flag to get confusion matrix.",
    )
    sub_parser_training.add_argument(
        "--stopwords_file_path",
        metavar="Stopwords File",
        type=str,
        help="Please provide file for stopwords.",
        default="",
        required=True,
        widget="FileChooser",
    )
    sub_parser_prediction = sub_parser.add_parser(
        "Prediction", parents=[parent_parser], help="args parse for cnn request"
    )
    sub_parser_prediction.set_defaults(pipeline_type="prediction")
    sub_parser_prediction.add_argument(
        "--input_file_path",
        metavar="Input File",
        type=str,
        help="Provide input file in csv format to get predictions",
        widget="FileChooser",
    )
    sub_parser_prediction.add_argument(
        "--output_file_path",
        metavar="Output Path",
        type=str,
        help="Provide output file directory.",
        widget="DirChooser",
    )
    sub_parser_prediction.add_argument(
        "--model_path",
        metavar="Model",
        type=str,
        help="Provide model's pickle file for prediction.",
        widget="FileChooser",
    )
    sub_parser_prediction.add_argument(
        "--test_input",
        metavar="Input Test Sentence",
        type=str,
        help="Provide input to be predicted.",
    )
    sub_parser_prediction.add_argument(
        "--stopwords_file_path",
        metavar="Stopwords File",
        type=str,
        help="Please provide file for stopwords.",
        default="",
        widget="FileChooser",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    feature_extractor = FeatureExtractor()
    if args.pipeline_type == "analysis":
        text_preprocessor = TextPreProcessor(
            stop_words_file_path=args.stopwords_file_path
        )
        analyser = DataAnalyser(
            input_file=args.input_file_path, text_preprocessor=text_preprocessor
        )
        analyser.get_data_distribution(plot_bar=args.plot_bar)
        analyser.get_word_weights(word_thresh=args.word_thresh)
        if args.word_cloud:
            analyser.generate_word_cloud()
    elif args.pipeline_type == "model_selection":
        text_preprocessor = TextPreProcessor(
            stop_words_file_path=args.stopwords_file_path
        )
        training_data_df = load_training_data(args.train_file_path)
        training_data_df["sentence"] = training_data_df["sentence"].map(
            text_preprocessor.process
        )
        features = feature_extractor.get_features_for_training(
            training_data_df["sentence"], args.vectorizer
        )
        labels = training_data_df["class"]
        apply_cross_validation(
            features=features,
            labels=labels,
            k_folds=args.kfolds,
            use_svm=args.use_svm,
            use_naive_bayes=args.use_naive_bayes,
            use_random_forest=args.use_random_forest,
            use_logistic_regression=args.use_logistic_regression,
            use_xgboost=args.use_xgboost,
            use_gradient_boosting=args.use_gradient_boosting,
            plot_cv_graph=True,
        )
    elif args.pipeline_type == "training":
        trainer = Trainer(
            train_file_path=args.train_file_path,
            val_file_path=args.val_file_path,
            stop_words_file_path=args.stopwords_file_path,
            model_name=args.best_model,
            feature_extractor=feature_extractor,
        )
        training_data_df = load_training_data(args.train_file_path)
        trainer.train(
            training_data_df,
            split_test_size=args.split_size,
            vectorizer_name=args.vectorizer,
            get_classification_report=args.get_classification_report,
            get_confusion_matrix=args.get_confusion_matrix,
        )
        validation_data_df = load_validation_data(args.val_file_path)
        trainer.validate(validation_data_df, vectorizer_name=args.vectorizer)
        if args.model_check_point_path:
            trainer.save_trained_model(args.model_check_point_path)
    elif args.pipeline_type == "prediction":
        if not args.stopwords_file_path:
            predictor = Predictor()
        else:
            predictor = Predictor(stop_words_file=args.stopwords_file_path)
        if args.input_file_path:
            predictor.predict_csv(
                args.input_file_path, args.output_file_path, args.model_path
            )
        if args.test_input:
            model, vectorizer = predictor.unpickle_the_model(args.model_path)
            predictor.predict(args.test_input, model, vectorizer)


if __name__ == "__main__":
    main()
