import pytest
from ojd_daps.flows.enrich.labs.skills.text_cleaning_utils import (
    clean_punctuation,
    remove_punctuation,
    pad_punctuation,
    unpad_punctuation,
    detect_sentences,
    lowercase,
    lemmatize_paragraph,
    remove_stopwords,
    clean_up,
    clean_text,
    clean_chunks,
    split_string,
    WordNetLemmatizer,
    lemmatise,
)

def test_WordNetLemmatizer():  # NB: WordNetLemmatizer is actually a local patch, not NLTK WNL
    assert WordNetLemmatizer() is WordNetLemmatizer()


def test_lemmatise():
    assert lemmatise("policies") == lemmatise("policy")

def test_clean_punctuation():
    assert clean_punctuation("• ‣ ◦ ⁃ ∙") == ", , , , ,"
    assert repr(clean_punctuation(r": / -")) == repr("     ")
    assert clean_punctuation(r"back\slash") == "back slash"
    assert (
        clean_punctuation("Preserves commas, semicollons; full stops.")
        == "Preserves commas, semicollons; full stops."
    )
    with pytest.raises(TypeError):
        clean_punctuation(123)


def test_remove_punctuation():
    assert repr(remove_punctuation("• ‣ ◦ ⁃ ∙")) == repr("         ")
    assert repr(remove_punctuation(r": / -")) == repr("     ")
    assert remove_punctuation(r"back\slash") == "back slash"
    assert (
        remove_punctuation("Removes commas, semicollons; full stops.")
        == "Removes commas  semicollons  full stops "
    )
    # Test special symbols that are preserved
    assert remove_punctuation(r"5+") == r"5+"
    assert remove_punctuation(r"C++") == r"C++"
    assert remove_punctuation(r"C#") == r"C#"
    assert remove_punctuation(r"C#") == r"C#"
    with pytest.raises(TypeError):
        clean_punctuation(123)


def test_pad_punctuation():
    assert (
        pad_punctuation(r"Pad around, most. of+ the; punctuation\ marks/")
        == r"Pad around ,  most .  of+ the ;  punctuation \  marks / "
    )
    with pytest.raises(TypeError):
        clean_punctuation(123)


def test_unpad_punctuation():
    assert (
        unpad_punctuation(r"Remove ; padding / around , punctation . marks")
        == r"Remove; padding/ around, punctation. marks"
    )
    with pytest.raises(TypeError):
        clean_punctuation(123)


def test_detect_sentences():
    assert (
        detect_sentences("Detect skillsAssess demand") == "Detect skills. Assess demand"
    )
    assert detect_sentences("USA") == "USA"
    with pytest.raises(TypeError):
        clean_punctuation(123)


def test_test_lowercase():
    assert lowercase("Detect skills. Assess demand") == "detect skills. assess demand"
    assert lowercase("USA") == "usa"
    with pytest.raises(TypeError):
        clean_punctuation(123)


def test_lemmatize_paragraph():
    assert lemmatize_paragraph("skills") == "skill"
    assert lemmatize_paragraph("skills.") == "skills."
    assert lemmatize_paragraph("Skills") == "Skills"
    with pytest.raises(TypeError):
        clean_punctuation(123)


def test_remove_stopwords():
    assert remove_stopwords("an apple and a tomato") == "apple tomato"
    with pytest.raises(TypeError):
        clean_punctuation(123)


def test_clean_up():
    assert clean_up("an     apple  and a  tomato  ") == "an apple and a tomato"
    assert clean_up(r"   ") == ""
    with pytest.raises(TypeError):
        clean_punctuation(123)


def test_clean_text():
    assert (
        clean_text("I went to the shop. I bought apples, oranges and a tomato")
        == "went shop bought apple orange tomato"
    )
    assert (
        clean_text(
            "I went to the shop. I bought apples, oranges and a tomato", keep_punct=True
        )
        == "went shop. bought apple, orange tomato"
    )
    with pytest.raises(TypeError):
        clean_punctuation(123)


def test_clean_chunks():
    assert clean_chunks("Apples, tomatos") == "apple tomato"
    assert clean_chunks(".NET") == "net"
    with pytest.raises(TypeError):
        clean_punctuation(123)


def test_split_string():
    assert split_string("First line\nSecond line") == ["First line", "Second line"]
    assert split_string("First sentence. Second sentence", separator=".") == [
        "First sentence",
        "Second sentence",
    ]
    assert split_string("First sentence.Second sentence", separator=".") == [
        "First sentence",
        "Second sentence",
    ]
    assert split_string(123) == []
