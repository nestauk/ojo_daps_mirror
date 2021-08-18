"""
skills.skills_detection_utils
--------------

Module to (1) generate phrases that serve as 'surface forms'
and have one-to-one relation to specific ESCO skills 'entities',
(2) to detect these surface forms in text, and (3) evaluate
surface form quality metrics.

"""
from functools import lru_cache
import boto3
import json
import pickle
from io import StringIO, BytesIO
import pandas as pd
import numpy as np
import re
import spacy
from collections import Counter
import os
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

# Custom modules
from .text_cleaning_utils import (
    clean_text,
    clean_chunks,
    clean_punctuation,
    split_string,
)
from .helper_utils import (
    save_to_s3,
    load_from_s3,
    save_json_to_s3,
)

# Parameters for the default spacy language model
DEF_LANGUAGE_MODEL = {"model": "en_core_web_sm", "disable": ["ner"]}
# Default filename patterns
DEF_SURFACE_FORMS_TABLE_FILENAME = "surface_forms_table_{}.csv"
DEF_MODEL_FILENAME = "surface_form_matcher_{}.pickle"
DEF_REMOVED_SURFACE_FORM_FILENAME = "surface_forms_removed_{}.json"
# Paths
SKILLS_LAB_DIR = Path(__file__).resolve().parents[0]

### SURFACE FORM GENERATION UTILS ###
def setup_spacy_model(model_parameters=DEF_LANGUAGE_MODEL):
    """
    Load and set up a spacy language model

    Args:
        model_parameters (dict): Dictionary containing language model parameters.
        The dictionary is expected to have the following structure:
            {
                "model":
                    Spacy language model name; for example "en_core_web_sm",
                "disable":
                    Pipeline components to disable for faster processing
                    (e.g. disable=["ner"] to disable named entity recognition).
                    If not required, set it to None.
            }

    Returns:
        (spacy.Language): A spacy language model
    """
    nlp = spacy.load(model_parameters["model"])
    if model_parameters["disable"] is not None:
        nlp.select_pipes(disable=model_parameters["disable"])
    return nlp


def flatten_skills_labels(entities, list_of_joined_labels, separator="\n"):
    """
    "Flattens" skills' alternate labels strings, and keeps track of the
    corresponding entity for each label

    Args:
        entities (list of int): Skill entity identifiers
        list_of_joined_labels (list of str): Skill entity labels, where each
            element corresponds to one entity, and the string might contain more
            than one label, separated by a separator symbol;
            For example: "use creative thinking\nthink creatively"

    Returns:
        flat_entities (list of int): Skill entity identifiers corresponding to each label
        flat_labels (list of str): Skill labels (one label per list element);
        For example: ["use creative thinking", "think creatively"]
    """
    labels_split = [split_string(s, separator=separator) for s in list_of_joined_labels]
    flat_labels = [
        label
        for list_of_entity_labels in labels_split
        for label in list_of_entity_labels
    ]
    flat_entities = [
        entities[i]
        for i, list_of_entity_labels in enumerate(labels_split)
        for label in list_of_entity_labels
    ]
    return flat_entities, flat_labels


def process_label_forms(entities, list_of_labels):
    """
    Returns surface forms derived from simple preprocessing of skill labels,
    together with their respective skills entity IDs

    Args:
        entities (list of int): Skill entity identifiers
        list_of_labels (list of str): Skills labels

    Returns:
        A list of tuples with skills entities and label-derived surface forms.
        For example:
            [(1, 'manage musical staff'), (2, 'perform planning')]
    """
    return list(zip(entities, [clean_text(s) for s in list_of_labels]))


def chunk_forms(texts, nlp, n_process=1):
    """
    Generate and process noun chunks from the provided texts

    Args:
        texts (list of str): List of input texts
        nlp (spacy.Language): Spacy language model
        n_process (int): Number of processors to use

    Yields:
        (list of str): Processed chunks for each input text string
    """
    texts = (clean_punctuation(s) for s in texts)
    docs = nlp.pipe(texts, batch_size=50, n_process=n_process)
    all_chunks = ([chunk.text for chunk in doc.noun_chunks] for doc in docs)
    return ([clean_chunks(s) for s in chunks] for chunks in all_chunks)


def process_chunk_forms(entities, texts, nlp, n_process=1):
    """
    Return surface forms from chunks, together with their respective entity IDs

    Args:
        entities (list of int): Skills entity identifiers
        texts (list of str): Skills-entity related texts
        nlp (spacy.Language): Spacy language model
        n_process (int): Number of processors to use

    Returns:
        A list of tuples with skills entities and chunk-derived surface forms.
        For example:
            [(1, 'musical staff'), (2, 'perform planning')]
    """
    list_of_lists_of_chunks = chunk_forms(texts, nlp, n_process=n_process)
    return [
        (entities[j], chunk)
        for j, chunks in enumerate(list_of_lists_of_chunks)
        for chunk in chunks
        if len(chunk) != 0
    ]


def create_surface_forms(entities, texts, chunk=False, nlp=None, n_process=1):
    """
    General function for producing surface forms, either by using noun chunking or not

    Args:
        entities (list of int): Skills entity identifiers
        texts (list of str): Skills-entity related texts
        is_chunk (boolean): If true, will produce noun chunks from texts
        nlp (spacy.Language): Spacy language model
        n_process (int): Number of processors to use
    """
    if not chunk:
        return process_label_forms(entities, texts)
    else:
        if nlp is None:
            nlp = setup_spacy_model()
        return process_chunk_forms(entities, texts, nlp=nlp, n_process=n_process)


def surface_form_dataframe(surface_form_dict):
    """
    Creates a pandas dataframe with surface forms

    Args:
        surface_form_dict (dict): Dictionary where key is the surface form type,
            and values are lists of tuples with skills entities and surface forms.
            For example:
            {'label_pref': [(1, 'manage musical staff'), (2, 'perform planning')],
             'chunk_pref': [(1, 'musical staff'), (2, 'planning')]}

    Returns:
        (pandas.DataFrame): Dataframe with columns for entity identifier, surface
            forms and surface form types.
    """
    surface_form_df = pd.DataFrame()
    for key in surface_form_dict.keys():
        df_temp = pd.DataFrame(
            surface_form_dict[key], columns=["entity", "surface_form"]
        )
        df_temp["surface_form_type"] = key
        surface_form_df = surface_form_df.append(df_temp, ignore_index=True)
    return surface_form_df


def deduplicate_surface_forms(surface_form_df, sort_order=None):
    """
    Sorting and deduplication of the surface form table. The sorting is necessary
    for removing the duplicated surface forms of less reliable types (e.g. if we
    find two identical forms but one is derived from a skills label, and the other one from
    chunking a description, we want to keep the label-derived surface form)

    Args:
        surface_form_df (pandas.DataFrame): Dataframe with columns for entity
            identifier, surface forms and surface form types.
        sort_order (list of str): Order in which the surface form types should
            be sorted. This should be in the order of importance/reliability.

    Returns:
        (pandas.DataFrame): Dataframe with sorted and deduplicated surface forms
    """
    # Use alphabetical sorting, if no sort_order is provided
    if sort_order is None:
        sorted(surface_form_df["surface_form_type"].unique())
    # Create a dataframe with surface forms
    df = (
        surface_form_df
        # Sort the surface forms in the provided order
        .assign(
            surface_form_type=lambda df: pd.Categorical(
                df.surface_form_type,
                categories=sort_order,
                ordered=True,
            )
        ).sort_values("surface_form_type")
        # Remove surface form duplicates within the same entity
        .drop_duplicates(["entity", "surface_form"], keep="first")
        # Remove surface form duplicates across different entities
        # Note: Other strategies for dealing with duplicate surface forms could be used;
        # for example, selecting the most relevant skill for each surface form
        .drop_duplicates("surface_form", keep="first")
    )
    return df


### REFINEMENT UTILS ###
def create_documents(lists_of_texts):
    """
    Create documents from lists of texts for further analysis, e.g. to
    calculate tf-idf scores of n-grams. For example:
        (['one','two'], ['cat', 'dogs']) -> ['one cat', 'two dogs']

    Args:
        lists_of_skill_texts (iterable of list of str): Contains lists of text
            features (e.g. label or description) to be joined up and processed to
            create the "documents"; i-th element of each list corresponds
            to the i-th entity/document

    Yields:
        (str): Created documents
    """
    # Check if all lists have the same length
    if len({len(i) for i in lists_of_texts}) == 1:
        # Transpose the lists of skill texts
        transposed_lists_of_texts = map(list, zip(*lists_of_texts))
        # Join up the skill texts for each skills entity
        return (
            " ".join(document_texts) for document_texts in transposed_lists_of_texts
        )
    else:
        raise ValueError("All lists in lists_of_texts should have the same length")


def tfidf_vectorise(documents):
    """
    Args:
        documents (list of str): Documents to vectorise

    Returns:
        tfidf_matrix (sparse numpy.matrix): Tf-idf-weighted document-term matrix with
            shape (n_documents, n_vocabulary_terms)
        vocabulary (dict): A mapping of terms to feature indices
    """
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1))
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    vocabulary = tfidf_vectorizer.vocabulary_
    return tfidf_matrix, vocabulary


def tfidf_representativity_mask(surface_form_df, tfidf_matrix, vocabulary):
    """
    For each surface form, estimate and compare how "representative" it is of each entity.

    Note that surface forms that are derived from labels (without chunking) are
    automatically assumed to be representative.

    NB: The assumptions for this step are quite ad-hoc

    Args:
        surface_form_df (pandas.DataFrame): Dataframe with columns for entity
            identifier, surface forms and surface form types.
        tfidf_matrix (sparse np.matrix): Tf-idf-weighted document-term matrix with
            shape (n_documents, n_vocabulary_terms)
        vocabulary (dict): A mapping of terms to feature indices

    Returns:
        sufficiently_representative (list of bool): True for rows of surface_form_df
            that will be kept, and False for frows that should be removed
    """
    # Threshold for which the score is assumed to be ambiguous (see below)
    AMBIG_THRESHOLD = 1

    sufficiently_representative = []
    # Check each surface form
    for j, row in surface_form_df.iterrows():
        if "label" in row.surface_form_type:
            # Don't evaluate surface forms derived from skill labels
            is_sufficient = True
        else:
            # Evaluate chunk form's representativeness to it's linked entity
            tf_idf_sums = tfidf_phrase_sums(row.surface_form, vocabulary, tfidf_matrix)
            best_matches, best_match_scores = find_best_n(tf_idf_sums, n=1)
            linked_entity_score = tf_idf_sums[row.entity]

            if row.entity == best_matches[0]:
                # If the best matching skill is also the surface form's entity
                is_sufficient = True
            elif (best_match_scores[0] > AMBIG_THRESHOLD) and (
                linked_entity_score > AMBIG_THRESHOLD
            ):
                # If the best match is ambiguous (because both best matching
                # entity and the linked entity have high scores)
                is_sufficient = True
            else:
                # Otherwise, not sufficiently representative
                is_sufficient = False
        sufficiently_representative.append(is_sufficient)
    return sufficiently_representative


def tfidf_phrase_sums(phrase, vocabulary, tfidf_matrix):
    """
    For the provided phrase, splits it into words (tokens) and, for each skills
    document, calculates the sum of the document's tf-idf vector elements that
    correspond to the words. This sum is assumed to indicate the phrase's
    representativeness of the skills document.

    Args:
        phrase (str): Text to check; usually a surface form
        tfidf_vectorizer (sklearn.feature_extraction.text.TfidfVectorizer):
            TfidfVectorizer holding transformed skills documents
        tfidf_matrix (numpy.ndarray): Matrix with rows corresponding to skills
            documents and columns to words (tokens)

    Returns:
        (list of float): Sums corresponding to each skills document

    """
    # Get individual tokens
    tokens = phrase.split(" ")
    # Get token indices
    token_indexes = []
    for token in tokens:
        try:
            token_indexes.append(vocabulary[token])
        except KeyError:
            pass
    phrase_tfidf_sums = (
        np.array(tfidf_matrix[:, token_indexes].sum(axis=1)).flatten().tolist()
    )
    return phrase_tfidf_sums


def find_best_n(list_of_numbers, n=1):
    """
    Return n largest elements and their indices from a given list

    Returns:
        best_n (numpy.ndarray): Indices of the largest n elements
        best_n_values (list of float): Sorted list of the largest n values
    """
    best_n = np.flip(np.argsort(list_of_numbers))[0:n]
    best_n_values = [list_of_numbers[x] for x in best_n]
    return best_n, best_n_values


def filter_rows(dataframe, rows_to_keep, forms_to_keep):
    """
    Helper function to filter surface forms while preserving
    the surface forms that have been selected manually

    Args:
        dataframe (pandas.DataFrame): Dataframe with surface forms
        rows_to_keep (list of bool): True for rows to keep, False for rows to discard
        forms_to_manually_keep: (list of str): Surface forms that have been manually selected to be preserved
    """
    rows_to_keep_manually = dataframe.surface_form.isin(forms_to_keep)
    rows_to_keep_series = pd.Series(rows_to_keep, index=dataframe.index)
    removed_forms = dataframe.loc[
        ~(rows_to_keep_series | rows_to_keep_manually)
    ].to_dict("records")
    new_dataframe = dataframe[(rows_to_keep) | (rows_to_keep_manually)]
    return new_dataframe, removed_forms


def remove_surface_forms(surface_form_df, manual_remove):
    """
    Remove manually rejected surface forms from the surface form dataframe

    Args:
        surface_form_df (pandas.DataFrame): Dataframe with columns for entity
            identifier, surface forms and surface form types.
        manual_assign (list of str): List specifying surface forms to keep

    Returns:
        (pandas.DataFrame): Dataframe with the new surface form and entity pairs

    """
    return surface_form_df[
        -surface_form_df.surface_form.isin(manual_remove)
    ]


def assign_surface_forms(surface_form_df, manual_assign):
    """
    Add manually assigned surface form to the surface form dataframe

    Args:
        surface_form_df (pandas.DataFrame): Dataframe with columns for entity
            identifier, surface forms and surface form types.
        manual_assign (list of dict): Dictionaries specifying surface forms and their new
            entity integer identifier

    Returns:
        (pandas.DataFrame): Dataframe with the new surface form and entity pairs

    """
    new_df = (
        surface_form_df
        # Add new assignments
        .append(manual_assign)
        # Remove the old entity assignments (if they exist)
        .drop_duplicates("surface_form", keep="last")
        # Take care of nulls
        .dropna(axis=0, subset=["surface_form"])
    )
    return new_df


def remove_and_reassign_forms(surface_form_df, manual_adjustments):
    """
    Combines remove_surface_forms() and assign_surface_forms()

    Args:
        surface_form_df (pandas.DataFrame): Dataframe with columns for entity
            identifier, surface forms and surface form types.
        manual_adjustemnts (dict): Dictionary specifying manual adjustments (cf.
            documentation of remove_surface_forms() and assign_surface_forms())

    Returns:
        (pandas.DataFrame): Dataframe with the new surface form and entity pairs

    """
    new_df = remove_surface_forms(surface_form_df, manual_adjustments["remove"])
    new_df = assign_surface_forms(new_df, manual_adjustments["assign"])
    return new_df


### PHRASE MATCHER and SKILLS DETECTION UTILS ###
def create_phrase_matcher(surface_forms, nlp):
    """
    Creates and returns a spacy phrase matcher

    Args:
        surface_forms (list of str): List of phrases for the phrase matcher
        nlp (spacy.Language): Spacy language model
    """
    matcher = spacy.matcher.PhraseMatcher(nlp.vocab)
    patterns = [nlp.make_doc(text) for text in surface_forms]
    matcher.add("TerminologyList", patterns)
    return matcher

def detect_skills(text, model, nlp, return_dict=False, debug=False):
    """
    Uses a spacy phrase matcher to detect skills

    Args:
        text (str): Text in which to detect skills; note that text should be preprocessed using clean_text() function
        model (dict): Dictionary with the spacy mather ('matcher'), surface
            forms dataframe ('surface_forms'), model name ('name') and spacy language
            model parameters for reference ('nlp')
        return_dict (bool): If True, outputs will be in a dict format
        debug (bool): If True, returns additional data associated with the surface forms

    Returns:
        (pandas.DataFrame or dict): Dataframe or dictionary with the detected surface forms, skills entities and their clusters

    """
    doc = nlp(text)
    matches = model["matcher"](doc)
    detected_forms = [str(doc[match[1] : match[2]]) for match in matches]
    rows = model["surface_forms"].surface_form.isin(detected_forms)
    results_dataframe = (model["surface_forms"][rows]
                         ).copy()
    if not debug:
        columns =[
            "is_predicted_OK",
            "surface_form",
            "surface_form_type",
            "preferred_label",
            "entity",
            "predicted_q",
            'cluster_0',
            'cluster_1',
            'cluster_2',
            'label_cluster_0',
            'label_cluster_1',
            'label_cluster_2',
            'manual_OK'
            ]
        results_dataframe = (results_dataframe[columns]
                             .sort_values('predicted_q')
                             .query('is_predicted_OK==1 | manual_OK==1')
                             .sort_values("surface_form_type", ascending=False)
                             .drop('is_predicted_OK', axis=1)
                             .drop('manual_OK', axis=1)
                             )

    if not return_dict:
        return results_dataframe
    else:
        return results_dataframe.to_dict("records")

### I/O UTILS ###
def save_model_locally(model, local_path=SKILLS_LAB_DIR/"models"):
    """
    Save the spacy matcher and associated data locally.

    Args:
        model (dict): Model dictionary, that should contain the following elements:
            'name': user-assigned name of the model,
            'matcher': spacy phrase matcher,
            'surface_forms': pandas.DataFrame with the matcher's phrases (surface forms)
                and their corresponding entity identifiers and preferred skill label, and
                the type of the surface form,
            'nlp': spacy language model parameters
        local_path (str): Path where to store the model
    """
    # Export the surface form table
    filename = DEF_SURFACE_FORMS_TABLE_FILENAME.format(model["name"])
    model["surface_forms"].to_csv(f"{local_path}/{filename}", index=False)
    # Export the full model dictionary
    filename = DEF_MODEL_FILENAME.format(model["name"])
    pickle.dump(model, open(f"{local_path}/{filename}", "wb"))


def save_model_in_s3(model):
    """
    Save the spacy matcher and associated in s3.

    Args:
        model (dict): Model dictionary, that should contain the following elements:
            'name': user-assigned name of the model,
            'matcher': spacy phrase matcher,
            'surface_forms': pandas.DataFrame with the matcher's phrases (surface forms)
                and their corresponding entity identifiers and preferred skill label, and
                the type of the surface form
            'nlp': spacy language model parameters
    """
    # Export the surface form table
    filename = DEF_SURFACE_FORMS_TABLE_FILENAME.format(model["name"])
    csv_buffer = StringIO()
    model["surface_forms"].to_csv(csv_buffer, index=False)
    save_to_s3(f"models/{filename}", csv_buffer.getvalue())
    # Export the full model dictionary
    filename = DEF_MODEL_FILENAME.format(model["name"])
    save_to_s3(f"models/{filename}", pickle.dumps(model))

@lru_cache()  # <--- Important
def load_model(model_name="", from_local=False, local_path=SKILLS_LAB_DIR/"models"):
    """
    Loads the pickled model

    Args:
        model_name (str): User-assigned name of the model
        from_local (bool): If True, the model will be loaded from the local storage
        local_path (str): Path where the local model is stored

    Returns:
        model (dict): Model dictionary, that should contain the following elements:
            'name': user-assigned name of the model,
            'matcher': spacy phrase matcher,
            'surface_forms': pandas.DataFrame with the matcher's phrases (surface forms)
                and their corresponding entity identifiers and preferred skill label, and
                the type of the surface form
            'nlp': spacy language model parameters
    """
    filename = DEF_MODEL_FILENAME.format(model_name)
    if from_local is False:
        model = pickle.loads(
            load_from_s3(f"models/{filename}")
        )
    else:
        model = pickle.load(
            open(f"{local_path}/{filename}", "rb")
        )
    # Ensure that entities are integers
    model['surface_forms'].entity = model['surface_forms'].entity.astype(int)
    return model

def save_removed_forms(
    removed_form_dict,
    model_name,
    local_path=SKILLS_LAB_DIR/"data/processed/surface_forms/removed",
    s3_copy=False,
):
    """
    Saves the dictionary with removed surface forms to a local path and optionally to s3

    Args:
        removed_form_dict (dict): A dict with forms that have been discarded,
            with keys indicating the step of the flow in which they were discarded
        model_name (str): User-assigned name of the model
        local_path (str): Path where to store the removed surface forms
        s3_copy (bool): If True, a copy of the dictionary will be saved on s3
    """
    filename = DEF_REMOVED_SURFACE_FORM_FILENAME.format(model_name)
    json.dump(removed_form_dict, open(f"{local_path}/{filename}", "w"), indent=4)
    if s3_copy:
        save_json_to_s3(
            removed_form_dict, f"data/processed/surface_forms/removed/{filename}"
        )


def load_removed_forms(model_name, local_path=SKILLS_LAB_DIR/"data/processed/surface_forms/removed"):
    """
    Loads in the surface forms that were removed when building the model.
    If such file is not found, an empty dictionary is returned.

    Args:
        model_name (str): User-assigned name of the model
        local_path (str): Path where the model is stored

    Returns:
        (dict): A dict with forms that have been discarded, with keys indicating
            the step of the flow in which they were discarded
    """
    filename = DEF_REMOVED_SURFACE_FORM_FILENAME.format(model_name)
    fpath = f"{local_path}/{filename}"
    # If there is no existent file associated with the model,
    # return an empty dictionary
    if not os.path.exists(fpath):
        return {}
    else:
        return json.load(open(fpath, "r"))

@lru_cache()
def regenerate_model(model_name='v02_1'):
    """ Rebuilds the skills detection components from the surface form table on S3 """
    # Load the surface form table
    csv_bytes = load_from_s3(f'models/surface_forms_table_{model_name}.csv')
    surface_form_table=pd.read_csv(BytesIO(csv_bytes))
    # Create the phrase matcher
    nlp = setup_spacy_model()
    matcher = create_phrase_matcher(surface_form_table.surface_form.to_list(), nlp)
    # Combine all components into the 'model'
    model = {
        'name': model_name,
        'surface_forms': surface_form_table,
        'matcher': matcher,
        'nlp': DEF_LANGUAGE_MODEL
    }
    return model

### SKILLS ANALYSIS UTILS ###
def count_surface_forms(detected_skills, key="surface_form"):
    """
    Counts skills attribute occurrences across the provided sample of job skills
    (by default, counts the occurrences of surface forms)

    Args:
        detected_skills (list of list of dict): Each nested list contains skills detected
            in a job description. Each skill is captured by a dictionary that has
            the following fields: "surface_form", "preferred_label", "entity", "surface_form_type"
        column (str): Determines which item to count

    Returns:
        counts (dict):  Dictionary of counts
    """
    counts = Counter(
        (skill[key] for job_skills in detected_skills for skill in job_skills)
    )
    return dict(counts)


def frequency_refinement(surface_form_df, counts, percentile_threshold=95):
    """
    Determine which surface forms might have to be removed based on their frequency
    (assuming that very frequently ocurring surface forms might be vauge or noisy terms)

    Args:
        surface_form_df (pandas.DataFrame): Dataframe with columns for entity identifier, surface
            forms and surface form types
        counts (dict):  Dictionary of surface form counts
        percentile_threshold (int or float): Percentile above which the surface
            forms will be considered for removal

    Returns:
        rows_to_keep (list of bool): True for rows of surface_form_df that will
            be kept, and False for frows that should be removed
    """
    # Threshold count above which the surface form will be considered for removal
    n_threshold = np.percentile(list(counts.values()), percentile_threshold)
    # Create a dataframe with surface forms, their type and counts
    surface_form_counts = pd.DataFrame(
        data={"surface_form": counts.keys(), "counts": counts.values()}
    ).merge(
        surface_form_df[["surface_form", "surface_form_type"]],
        on="surface_form",
        how="left",
        validate="1:1",
    )
    forms_to_remove = surface_form_counts[
        # Remove surface forms that are above the frequency threshold,
        (surface_form_counts.counts > n_threshold)
        # and that consist of only one word
        & (surface_form_counts.surface_form.apply(lambda x: len(x.split(" ")) <= 1))
        # and that are not derived from the preferred label
        & (surface_form_counts.surface_form_type != "label_pref")
    ].surface_form.to_list()
    rows_to_keep = (surface_form_df.surface_form.isin(forms_to_remove)==False).to_list()
    return rows_to_keep
