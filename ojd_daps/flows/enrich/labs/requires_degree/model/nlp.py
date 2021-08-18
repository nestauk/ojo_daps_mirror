"""Text processing support functions for requires_degree regex model."""
import re


def regex_model(regex):
    """Returns a function that returns true if the job description includes a degree requirement."""
    re_ = re.compile(regex)
    return lambda description: any(re_.findall(description))


def strip_last_term(text):
    """Strip off the last term, as it may have been truncated"""
    last_space_idx = text.rfind(" ")
    if last_space_idx > 0:  # is -1 in the case of no spaces
        return text[:last_space_idx]
    return text


def remove_outer_brackets(text):
    """Reed descriptions (currently) start/end with brackets"""
    if text.startswith("["):
        text = text[1:]
    if text.endswith("]"):
        text = text[:-1]
    return text.strip()


def clean_description(text):
    """Apply cleaning steps to the description"""
    text = remove_outer_brackets(text)
    text = strip_last_term(text)
    return text
