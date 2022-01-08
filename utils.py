import os
import collections
import itertools
from typing import Any, Iterable, Callable, List
import pandas as pd
import numpy as np
from spacy.language import Language
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn import preprocessing


def load_data(input_path: str) -> pd.DataFrame:
    """Loads data from .txt files and returns dataframe with Text and Label columns"""
    data = []
    for path, sub_dirs, files in os.walk(input_path):
        for name in files:
            with open(os.path.join(path, name), 'r', encoding='utf-8') as text_file:
                text = text_file.read()
                data.append((path.split('/')[-1], text))
    return pd.DataFrame(data, columns=['Label', 'Text'])


def apply_many_functions(input_object: Iterable[Any], function: Callable[[Any], Any],
                         *functions: Callable[[Any], Any]) -> List[Any]:
    """Applies several functions to input_object"""
    if functions:
        return apply_many_functions(map(function, input_object), *functions)
    return list(map(function, input_object))


def count_word_freq(list_text: list) -> dict:
    """Counts word frequency in list of texts"""
    text_split = [text.split() for text in list_text]
    word_freq_dict = dict(collections.Counter(list(itertools.chain(*text_split))))
    return word_freq_dict


def text_to_keras_sequence(tokenizer_object: Tokenizer, list_text: list):
    """Transforms list of texts to keras padded sequences"""
    tokenizer_object.fit_on_texts(list_text)
    sequences = tokenizer_object.texts_to_sequences(list_text)
    sequences_pad = pad_sequences(sequences, padding='post')
    return sequences_pad


def create_spacy_embedding_matrix(tokenizer_object: Tokenizer,
                                  spacy_nlp_object: Language) -> np.ndarray:
    """Creates embedding matrix using pretrained SpaCy's embedding vectors"""
    embedding_matrix = np.zeros((tokenizer_object.num_words, 300))

    for word, i in tokenizer_object.word_index.items():
        if i < tokenizer_object.num_words:
            embedding_vector = spacy_nlp_object(word).vector
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        else:
            break
    return embedding_matrix


def transform_dependent_variable(labels: list) -> np.ndarray:
    """Transforms labels vector to categorical data matrix"""
    le = preprocessing.LabelEncoder()
    label_encoded = le.fit_transform(labels)
    categorical_labels = to_categorical(label_encoded, dtype="uint8")
    return categorical_labels
