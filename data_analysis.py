import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import lower_case_comment
from wordcloud import WordCloud, STOPWORDS


class DataAnalyser:
    def __init__(self, input_file, text_preprocessor):
        self.data_frame_file = pd.read_csv(input_file)
        self.text_preprocessor = text_preprocessor

    def get_data_distribution(self, plot_bar):
        print(
            f" Total Records: {len(self.data_frame_file)}",
            "\n",
            f"Number of classes: {len(self.data_frame_file['class'].unique())}",
            "\n",
            f"Number of records per class:\n {pd.value_counts(self.data_frame_file['class'])}",
        )
        if plot_bar:
            plt.figure(figsize=(8, 6))
            pd.value_counts(self.data_frame_file["class"]).plot.bar(fontsize=9)
            plt.show()

    def get_word_weights(self, word_thresh, plot_word_graph=False):
        merged_sentences = " ".join(
            " ".join(word.lower() for word in set(sentence.split(" ")))
            for sentence in self.data_frame_file["sentence"]
        )
        pre_processed_merged_sentences = lower_case_comment(merged_sentences)

        phrases = self.text_preprocessor.rake_obj.get_scores(
            pre_processed_merged_sentences
        )
        phrase_list, weight_list = list(), list()
        for i in range(word_thresh):
            phrase_list.append(phrases[i][0])
            weight_list.append(phrases[i][1])
            print(f"Phrase: {phrases[i][0]} Phrase Weight: {phrases[i][1]}")
        if plot_word_graph:
            pass

    def generate_word_cloud(self):
        if self.text_preprocessor.stop_words_file_path:
            stopwords_set = self.text_preprocessor.read_stopwords_in_set()
        else:
            stopwords_set = STOPWORDS
        word_cloud = WordCloud(
            stopwords=stopwords_set,
            background_color="black",
            width=1200,
            height=1000,
            max_words=1000,
        ).generate(
            " ".join(
                " ".join(word.lower() for word in set(sentence.split(" ")))
                for sentence in self.data_frame_file["sentence"]
            )
        )
        plt.imshow(word_cloud)
        plt.axis("off")
        plt.show()


# if __name__ == "__main__":
#     from preprocessing import TextPreProcessor
#
#     a = DataAnalyser(
#         input_file=r"D:\GitHub\text-analysis-and-classification-gui-tool\example_use_case\data\fake_news_valid.csv",
#         text_preprocessor=TextPreProcessor(
#             stop_words_file_path=r"D:\GitHub\text-analysis-and-classification-gui-tool\example_use_case\stopwords\stopwords_list.txt"
#         ),
#     )
#     a.generate_word_cloud()
#     a.get_data_distribution(plot_bar=True)
#     a.get_word_weights(word_thresh=100)
