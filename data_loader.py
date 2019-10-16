import pandas as pd


def load_training_data(training_file):
    training_file_df = pd.read_csv(training_file, encoding='utf-8')
    training_file_df = training_file_df.sample(frac=1, replace=False)
    training_file_df = training_file_df[pd.notnull(training_file_df["sentence"])]
    training_file_df["class_id"] = training_file_df["class"].factorize()[0]
    return training_file_df


def load_validation_data(validation_file):
    validation_file_df = pd.read_csv(validation_file, encoding="UTF-8")
    validation_file_df = validation_file_df.sample(frac=1, replace=False)
    validation_file_df = validation_file_df[pd.notnull(validation_file_df["sentence"])]
    validation_file_df["class_id"] = validation_file_df["class"].factorize()[0]
    return validation_file_df
