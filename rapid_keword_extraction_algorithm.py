# !/usr/bin/env python

# title           :SmartRake.py
# author          :manedesh
# usage           :use during text pre-processing to extract keywords

import re


def is_number(s):
    try:
        float(s) if "." in s else int(s)
        return True
    except ValueError:
        return False


def load_the_stop_words(stop_word_file):
    stop_words = []
    for line in open(stop_word_file):
        if line.strip()[0:1] != "#":
            for word in line.split():
                stop_words.append(word)
    return stop_words


def separate_the_words(text, min_word_return_size):
    splitter = re.compile("[^a-zA-Z0-9_\\+\\-/]")
    words = []
    for single_word in splitter.split(text):
        current_word = single_word.strip().lower()
        if (
            len(current_word) > min_word_return_size
            and current_word != ""
            and not is_number(current_word)
        ):
            words.append(current_word)
    return words


def split_sentences(text):
    sentence_delimiters = re.compile(
        u"[.!?,;:_\t\\\\\"\\(\\)\\'\u2019\u2013]|\\s\\-\\s"
    )  # delimiters except hyphen
    sentences = sentence_delimiters.split(text)
    return sentences


def build_stop_word_regex(stop_word_file_path):
    stop_word_list = load_the_stop_words(stop_word_file_path)
    stop_word_regex_list = []
    for word in stop_word_list:
        word_regex = r"\b" + word + r"(?![\w-])"  # look ahead for hyphen
        stop_word_regex_list.append(word_regex)
    stop_word_pattern = re.compile("|".join(stop_word_regex_list), re.IGNORECASE)
    # print(stop_word_pattern)
    return stop_word_pattern


def generate_candidate_keywords(sentence_list, stopword_pattern):
    phrase_list = []
    for s in sentence_list:
        tmp = re.sub(stopword_pattern, "|", s.strip())
        phrases = tmp.split("|")
        for phrase in phrases:
            phrase = phrase.strip().lower()
            if phrase != "":
                phrase_list.append(phrase)
    return phrase_list


def calculate_word_scores(phrase_list):
    word_frequency = {}
    word_degree = {}
    for phrase in phrase_list:
        word_list = separate_the_words(phrase, 0)
        word_list_length = len(word_list)
        word_list_degree = word_list_length - 1
        # if word_list_degree > 3: word_list_degree = 3
        for word in word_list:
            word_frequency.setdefault(word, 0)
            word_frequency[word] += 1
            word_degree.setdefault(word, 0)
            word_degree[word] += word_list_degree
            # word_degree[word] += 1/(word_list_length*1.0)
    for item in word_frequency:
        word_degree[item] = word_degree[item] + word_frequency[item]

    # Calculate Word scores = deg(w)/frew(w) (formula)
    word_score = {}
    for item in word_frequency:
        word_score.setdefault(item, 0)
        word_score[item] = word_degree[item] / (word_frequency[item] * 1.0)
    # word_score[item] = word_frequency[item]/(word_degree[item] * 1.0)
    return word_score


def generate_candidate_keyword_scores(phrase_list, word_score):
    keyword_candidates = {}
    for phrase in phrase_list:
        keyword_candidates.setdefault(phrase, 0)
        word_list = separate_the_words(phrase, 0)
        candidate_score = 0
        for word in word_list:
            candidate_score += word_score[word]
        keyword_candidates[phrase] = candidate_score
    return keyword_candidates


class Rake(object):
    """
    Rapid Automatic Keywords Extraction Algorithm Implementation
    Utility class to get the candidate keywords from the text and rank them according to their weights
    """

    # get the stopwords and build regex pattern
    def __init__(self, stop_words_path):
        self.stop_words_path = stop_words_path
        self.__stop_words_pattern = build_stop_word_regex(stop_words_path)

    def run(self, text):
        sentence_list = split_sentences(text)

        phrase_list = generate_candidate_keywords(
            sentence_list, self.__stop_words_pattern
        )
        # print(phrase_list)

        word_scores = calculate_word_scores(phrase_list)
        # print(word_scores)

        keyword_candidates = generate_candidate_keyword_scores(phrase_list, word_scores)
        # print("Type: keyword_candidates : ", type(keyword_candidates))
        # print(keyword_candidates)

        # sorted_keywords = sorted(keyword_candidates.items(), key=operator.itemgetter(1), reverse=True)
        # print(sorted_keywords)
        sorted_keywords = keyword_candidates.items()

        return sorted_keywords
