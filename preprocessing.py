import nltk
import re
import time
import unicodedata
import spacy
import string
from collections import OrderedDict
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from contractions import contractions_dict
from string import digits
from nltk.corpus import brown
from nltk.corpus import wordnet as wn
from nltk.stem import LancasterStemmer, SnowballStemmer, PorterStemmer
from autocorrect import spell
from abbreviations import synonyms, abbreviations_map
from rapid_keword_extraction_algorithm import Rake

meaningful_words_lower_cased = set(i.lower() for i in brown.words())


def strip_html_tags(text: str):
    try:
        soup = BeautifulSoup(text, "html.parser")
        stripped_text = soup.get_text()
    except Exception as e:
        print(f"Exception in HTML parsing: {e}")
        return text
    return stripped_text


def remove_accented_chars(text: str):
    try:
        text = (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore")
            .decode("utf-8", "ignore")
        )
    except Exception as e:
        print(f"Exception in removing accented characters: {e}")
        return text
    return text


def replace_abbreviations(comment: str):
    words = comment.split()
    line = []
    for word in words:
        if word.lower() in abbreviations_map:
            # get abbreviation from dictionary
            line.append(abbreviations_map[word.lower()])
        else:
            line.append(word)  # append original word otherwise
    return " ".join(word for word in line)


def expand_contractions(text: str, contraction_mapping=contractions_dict):
    contractions_pattern = re.compile(
        "({})".format("|".join(contraction_mapping.keys())),
        flags=re.IGNORECASE | re.DOTALL,
    )

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = (
            contraction_mapping.get(match)
            if contraction_mapping.get(match)
            else contraction_mapping.get(match.lower())
        )
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    try:
        expanded_text = contractions_pattern.sub(expand_match, text)
        expanded_text = re.sub("'", "", expanded_text)
    except Exception as e:
        print(f"Exception in expanding contractions: {e}")
        return text
    return expanded_text


def remove_url(comment: str):
    patterns = [r"http\S+", r"www\S+"]
    out = comment
    for pattern in patterns:
        out = re.sub(pattern, "", out)
    return out


def tokenize_comment(comment: str):
    tokens = nltk.word_tokenize(comment)
    return " ".join(token for token in tokens)


def strip_punctuation(comment: str):
    regex = re.compile("[%s]" % re.escape(string.punctuation))
    comment = regex.sub(" ", comment)
    return comment


def lower_case_comment(comment: str):
    return comment.lower()


def remove_numbers_from_comment(comment: str):
    remove_digits = str.maketrans("", "", digits)
    return comment.translate(remove_digits)


def get_rake_object(stopwordfile: str):
    stop_words_path = "..\\Stopwords\\"
    return Rake(stop_words_path + stopwordfile)


def auto_correct_spell(comment: str):
    comment = comment.split()
    return " ".join(spell(word) for word in comment)


def extract_meaningful_words_test(comment: str):
    comment_words = comment.split()
    filtered_comment = [
        word for word in comment_words if word.lower() in meaningful_words_lower_cased
    ]
    return " ".join(word for word in filtered_comment)


def extract_meaningful_words(comment: str):
    try:
        comment_words = comment.split()
        filtered_comment = [word for word in comment_words if wn.synsets(word.lower())]
    except Exception as e:
        print(f"Exception in extracting meaningful words: {e}")
        return comment
    return " ".join(word for word in filtered_comment)


def replace_synonyms(comment: str):
    comment = comment.split()
    new_str = " "
    for word in comment:
        if word.lower() in synonyms:
            new_str += synonyms[word.lower()] + " "
        else:
            new_str += word + " "
    return new_str


def simple_stemming(text: str):
    stemmer = LancasterStemmer()
    text = " ".join([stemmer.stem(word.lower()) for word in text.split()])
    return text


def porter_stemming(text: str):
    stemmer = PorterStemmer()
    text = " ".join([stemmer.stem(word.lower()) for word in text.split()])
    return text


def snowball_stemming(text: str):
    stemmer = SnowballStemmer("english")
    text = " ".join([stemmer.stem(word.lower()) for word in text.split()])
    return text


def lemmatize_text(text: str):
    nlp = spacy.load("en", parse=True, tag=True, entity=True)
    text = nlp(text)
    text = " ".join(
        [word.lemma_ if word.lemma_ != "-PRON-" else word.text for word in text]
    )
    return text


def remove_stop_words(comment: str):
    stop_words = set(stopwords.words("english"))
    word_tokens = comment.split()
    filtered_comment = " ".join(w for w in word_tokens if w not in stop_words)
    return filtered_comment


def remove_special_characters(text: str, remove_digits=False):
    pattern = r"[^a-zA-z0-9\s]" if not remove_digits else r"[^a-zA-z\s]"
    text = re.sub(pattern, "", text)
    return text


def pre_process_for_ad(comment):
    tt = time.time()

    comment = remove_stop_words(comment)
    comment = strip_html_tags(comment)
    comment = remove_accented_chars(comment)
    comment = remove_numbers_from_comment(comment)
    comment = expand_contractions(comment)
    comment = strip_punctuation(comment)
    comment = remove_url(comment)
    comment = tokenize_comment(comment)
    comment = lower_case_comment(comment)
    comment = extract_meaningful_words(comment)
    comment = simple_stemming(comment)

    print(f"Pre-processing time:\t{time.time()-tt}")
    return comment


class TextPreProcessor:
    def __init__(self, stop_words_file_path):
        self.stop_words_file_path = stop_words_file_path
        self.preprocessing_step_dict = OrderedDict({})
        self.rake_obj = Rake(self.stop_words_file_path)

    def extract_keywords_using_rake(self, comment):
        keywords = self.rake_obj.run(comment)
        return " ".join(str(word[0]) for word in keywords)

    def process(self, comment):
        self.preprocessing_step_dict["comment"] = comment

        # function call to remove 'HTML' tags from the comment
        comment = strip_html_tags(comment)

        # function call to remove accented characters from the comment
        comment = remove_accented_chars(comment)

        # function call to remove HTML tags from the comment
        comment = remove_numbers_from_comment(comment)

        # expands remained contraction or abbreviations if any
        comment = expand_contractions(comment)

        # function call to remove punctuations from the comment
        comment = strip_punctuation(comment)
        self.preprocessing_step_dict["noise_removed"] = comment

        # function call to replace abbreviations in the comment
        comment = replace_abbreviations(comment)
        self.preprocessing_step_dict["replace_abbr"] = comment

        # function call to remove URL from the comment
        comment = remove_url(comment)

        # tokenization
        comment = tokenize_comment(comment)

        # lowercase
        comment = lower_case_comment(comment)

        # perform Rapid keywords extraction algorithm
        comment = self.extract_keywords_using_rake(comment)
        self.preprocessing_step_dict["rake_output"] = comment

        # extract only essential words
        comment = extract_meaningful_words(comment)
        self.preprocessing_step_dict["meaningful_words"] = comment

        # Replace synonyms with root word
        comment = replace_synonyms(comment)

        # stem the comment
        comment = simple_stemming(comment)
        self.preprocessing_step_dict["stemmed_out"] = comment

        return comment

    def get_pre_processing_steps(self):
        self.process()
        return self.preprocessing_step_dict
