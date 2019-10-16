import pickle
import pandas as pd
from os import path
from preprocessing import TextPreProcessor


class Predictor:
    def __init__(self, stop_words_file):
        self.preprocessor = TextPreProcessor(stop_words_file_path=stop_words_file)

    @staticmethod
    def unpickle_the_model(model_pickle_file):
        svc_model_pkl = open(model_pickle_file, "rb")
        model = pickle.load(svc_model_pkl)
        vectorizer = pickle.load(svc_model_pkl)
        return model, vectorizer

    def predict(self, sentence, model, vectorizer):
        if not sentence:
            return "None"
        pre_processed_comment = self.preprocessor.process(sentence)
        features = vectorizer.transform([pre_processed_comment])
        predicted_type = model.predict(features)
        print(f"sentence: {sentence} Prediction output type:{predicted_type}")
        return predicted_type

    def predict_csv(self, input_file, out_put_file_path, model_path):
        input_file_df = pd.read_csv(input_file)
        out_df_col = ["class", "output class", "sentence", "preprocessed sentence", "ID"]
        output_df = pd.DataFrame(columns=out_df_col)
        output_df["class"] = input_file_df["class"]
        output_df["sentence"] = input_file_df["sentence"]
        output_df["preprocessed sentence"] = output_df["sentence"].map(
            self.preprocessor.process
        )
        output_df["ID"] = input_file_df["ID"]
        model, vectorizer = self.unpickle_the_model(model_path)
        out_put_file_name = "Predicted"+path.basename(input_file)
        output_types = list()
        for comment in output_df["preprocessed sentence"].values:
            out_report_type = self.predict(comment, model, vectorizer)
            output_types.append(out_report_type)
        output_df["output class"] = pd.Series(output_types)
        output_df.to_csv(path.join(out_put_file_path, out_put_file_name))
        print("File predicted successfully!")