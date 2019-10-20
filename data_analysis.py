import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS


class DataAnalyser:
    def __init__(self, input_file, stopwords_file):
        self.data_frame_file = pd.read_csv(input_file)
        self.stopwords_file = stopwords_file

    def read_stopwords_in_set(self):
        stopwords = set()
        try:
            with open(self.stopwords_file, "rb") as file:
                # stopwords_file must a contain single stop-word per line
                line = file.readline()
                while line:
                    stopwords.add(line.strip())
                    line = file.readline()
        except FileNotFoundError as fe:
            print(f"Stopwords file not found!{fe}")
        return stopwords

    def count_records(self):
        return len(self.data_frame_file)

    def get_data_distribution(self):
        count_df = self.data_frame_file.groupby(["class"]).count()[["ID"]]
        sns.barplot(x=count_df.Type, y=count_df.AnomalyID)

    def plot_bar_x(self, classes, count):
        raise NotImplementedError

    def generate_word_cloud(self):
        if self.stopwords_file:
            stopwords_set = self.read_stopwords_in_set()
        else:
            stopwords_set = STOPWORDS
        word_cloud = WordCloud(
            stopwords=stopwords_set, background_color="black", width=1200, height=1000
        ).generate(
            " ".join(
                " ".join(word.lower() for word in set(sentence.split(" ")))
                for sentence in self.data_frame_file["sentence"]
            )
        )
        plt.imshow(word_cloud)
        plt.axis("off")
        plt.show()

    def plot_word_weights(self):
        raise NotImplementedError


# if __name__ == "__main__":
#     a = DataAnalyser(
#         input_file=r"D:\GitHub\text-analysis-and-classification-gui-tool\example_use_case\data\fake_news_valid.csv",
#         stopwords_file=r"D:\GitHub\text-analysis-and-classification-gui-tool\example_use_case\stopwords\stopwords_list.txt",
#     )
#     a.generate_word_cloud()
