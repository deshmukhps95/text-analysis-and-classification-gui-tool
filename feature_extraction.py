from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer,
    HashingVectorizer,
)


class FeatureExtractor:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            min_df=1,
            norm="l2",
            encoding="utf-8",
            ngram_range=(1, 3),
            stop_words=None,
            max_features=None,
            binary=False,
        )
        self.count_vectorizer = CountVectorizer(
            encoding="UTF-8",
            decode_error="ignore",
            analyzer="word",
            ngram_range=(1, 3),
            max_features=None,
            binary=False,
        )
        self.hash_vectorizer = HashingVectorizer(
            encoding="UTF-8",
            decode_error="ignore",
            analyzer="word",
            ngram_range=(1, 3),
            n_features=2 ** 15,
            binary=True,
        )
        self.vectorizer_dict = {
            "TF-IDF": self.tfidf_vectorizer,
            "Count": self.count_vectorizer,
            "Hash": self.hash_vectorizer,
        }

    def get_features_for_training(self, document_frame, vectorizer_name):
        return (
            self.vectorizer_dict[vectorizer_name].fit_transform(document_frame).toarray()
        )

    def get_features_for_testing(self, document_frame, vectorizer_name):
        return self.vectorizer_dict[vectorizer_name].transform(document_frame).toarray()
