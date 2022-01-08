import string
import re
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)


class CleanText:
    """Cleans text and prepareit for modelling"""

    def __init__(self, input_text: str):
        self.input_text = input_text

    def remove_punctuations(self) -> str:
        """Removes punctuations from the text"""
        remove_punctuations_map = dict.fromkeys(map(ord, string.punctuation))
        return self.input_text.translate(remove_punctuations_map)

    def remove_stop_words(self) -> str:
        """Removes English stop words from the text"""
        stop_words = stopwords.words('english')
        return " ".join([word for word in self.input_text.split() if word not in stop_words])

    def remove_digits(self) -> str:
        """Removes digits from the text"""
        remove_digits_map = dict.fromkeys(map(ord, string.digits))
        return self.input_text.translate(remove_digits_map)

    def remove_empty_lines(self) -> str:
        """Removes empty/blank lines from the text"""
        return "".join(filter(str.strip, self.input_text.splitlines(True)))

    def remove_whitespaces(self) -> str:
        """Removes whitespaces from the text"""
        return re.sub(r'\s\s+', ' ', self.input_text.strip())

    @staticmethod
    def get_wordnet_pos(word):
        """Maps nltk pos tagger to word"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)

    def lemmatize_words(self) -> str:
        """Transforms words in text to their base forms"""
        lemmatizer = WordNetLemmatizer()
        return " ".join([lemmatizer.lemmatize(word, self.get_wordnet_pos(word))
                         for word in self.input_text.split()])
