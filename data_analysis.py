import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class DataAnalyser:
    def __init__(self, input_file):
        self.data_frame_file = pd.read_csv(input_file)
        print(self.data_frame_file["Type"])

    def count_records(self):
        return len(self.data_frame_file)

    def get_data_distribution(self):
        count_df = self.data_frame_file.groupby(["Type"]).count()[["AnomalyID"]]
        sns.barplot(x=count_df.Type, y=count_df.AnomalyID)

    def plot_bar_x(self, classes, count):
        pass

    def generate_word_cloud(self):
        pass

    def plot_word_weights(self):
        pass
