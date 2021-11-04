"""
skills.text_cleaning_utils
--------------

Module to for preprocessing and online job vacancy and skills-related text data.

"""

from toolz import pipe
import re
import nltk
from functools import lru_cache

### Compiling regex patterns as they might get used many times over ###

# Hardcoded rules for dealing with punctuation marks and other custom symbols
punctuation_replacement_rules = {
    # old patterns: replacement pattern
    "[\u2022,\u2023,\u25E6,\u2043,\u2219]": ",",  # Convert bullet points to commas
    r"[-/:\\]": " ",  # Convert colon, hyphens and forward and backward slashes to spaces
    r"[^a-zA-Z0-9,.; #(++)]": "",  # Preserve spaces, commas, full stops, semicollons for discerning noun chunks
}

# Patterns for cleaning punctuation, for clean_punctuation()
compiled_punct_patterns = [re.compile(p) for p in punctuation_replacement_rules.keys()]
punct_replacement = list(punctuation_replacement_rules.values())

# Pattern for fixing a missing space between enumerations, for split_sentences()
compiled_missing_space_pattern = re.compile("([a-z])([A-Z])([a-z])")

compiled_nonalphabet_nonnumeric_pattern = re.compile(r"([^a-zA-Z0-9 #(++)+])")
compiled_padded_punctuation_pattern = re.compile(r"( )([^a-zA-Z0-9 #(++)+])")

### Components of the text preprocessing pipeline ###
@lru_cache()
def WordNetLemmatizer():
    nltk.download("wordnet")
    return nltk.WordNetLemmatizer()


@lru_cache()
def get_stopwords():
    """Returns the set of English stopwords from nltk"""
    nltk.download("stopwords")
    from nltk.corpus import stopwords

    return set(stopwords.words("english"))


def lemmatise(term):
    """Apply the NLTK WN Lemmatizer to the term"""
    lem = WordNetLemmatizer()
    return lem.lemmatize(term)


def clean_punctuation(text):
    """Replaces punctuation according to the predefined patterns"""
    for j, pattern in enumerate(compiled_punct_patterns):
        text = pattern.sub(punct_replacement[j], text)
    return text


def remove_punctuation(text):
    """Remove punctuation marks and replace with spaces (to facilitate lemmatisation)"""
    text = compiled_nonalphabet_nonnumeric_pattern.sub(r" ", text)
    return text


def pad_punctuation(text):
    """Pad punctuation marks with spaces (to facilitate lemmatisation)"""
    text = compiled_nonalphabet_nonnumeric_pattern.sub(r" \1 ", text)
    return text


def unpad_punctuation(text):
    """Remove spaces preceding punctuation marks"""
    text = compiled_padded_punctuation_pattern.sub(r"\2", text)
    return text


def detect_sentences(text):
    """
    Splits a word written in camel-case into separate sentences. This fixes a case
    when the last word of a sentence in not seperated from the capitalised word of
    the next sentence. This tends to occur with enumerations.

    For example, the string "skillsBe" will be converted to "skills. Be"

    Note that the present solution doesn't catch all such cases (e.g. "UKSkills")

    Reference: https://stackoverflow.com/questions/1097901/regular-expression-split-string-by-capital-letter-but-ignore-tla
    """
    text = compiled_missing_space_pattern.sub(r"\1. \2\3", text)
    return text


def lowercase(text):
    """Converts all text to lowercase"""
    return text.lower()


def lemmatize_paragraph(text):
    """
    Lemmatizes each word in a paragraph.
    Note that this function has to be included in a processing pipeline as, on
    its own, it does not deal with punctuation marks or capital letters.
    """
    # Lemmatize each word
    text = " ".join([lemmatise(token) for token in text.split(" ")])
    return text


def remove_stopwords(text):
    """Removes stopwords"""
    stopwords = get_stopwords()
    text = " ".join(filter(lambda token: token not in stopwords, text.split(" ")))
    return text


def clean_up(text):
    """Removes extra spaces between words"""
    text = " ".join(text.split()).strip()
    return text


def clean_text(text, keep_punct=False):
    """
    Pipeline for preprocessing online job vacancy and skills-related text.
    NB: If 'keep_punct' is True, then commas, full stops and semicollons are preserved

    Args:
        text (str): Text to be processed via the pipeline
    """
    if keep_punct is False:
        return pipe(
            text,
            detect_sentences,
            lowercase,
            remove_punctuation,
            lemmatize_paragraph,
            remove_stopwords,
            clean_up,
        )
    elif keep_punct is True:
        return pipe(
            text,
            detect_sentences,
            lowercase,
            clean_punctuation,
            pad_punctuation,
            lemmatize_paragraph,
            remove_stopwords,
            unpad_punctuation,
            clean_up,
        )


def clean_chunks(text):
    """Pipeline for processing noun chunks"""
    return pipe(
        text,
        remove_punctuation,
        lowercase,
        lemmatize_paragraph,
        remove_stopwords,
        clean_up,
    )


def split_string(string, separator="\n"):
    """Checks if input is a string and then splits and strips it"""
    if type(string) is str:
        return [s.strip() for s in string.split(separator)]
    else:
        return []
