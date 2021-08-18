# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Evaluation of surface form quality

# %%
import logging
import pandas as pd
import numpy as np
from collections import Counter
import os
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial.distance import cdist
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

from ojd_daps.flows.enrich.labs.skills.helper_utils import DATA_PATH, get_esco_skills, get_skill_embeddings, pickle_model
import ojd_daps.flows.enrich.labs.skills.skills_detection_utils as skills_detection_utils
from ojd_daps.flows.enrich.labs.skills.skills_detection_utils import (
    clean_text,
    split_string,
    create_documents,
    tfidf_vectorise,
    load_model,
    save_model_locally,
    save_model_in_s3
)

# Path to files from manual review
MANUAL_CHECK_DIR = DATA_PATH / 'aux/manual_checking'

# Transformer model
BERT_MODEL = 'paraphrase-distilroberta-base-v1'
BERT_TRANSFORMER = SentenceTransformer(BERT_MODEL)

# Import the latest skills detection model ('v01_1')
skills_model = load_model('v01_1', from_local=True)

# %%
# Parameters for the ML model
NUMERICAL_FEATURES = [
    'score_emb_similarity',
    'score_tfidf_sum',
    'score_tfidf_mean',
    'score_idf_sum',
    'score_idf_mean'
]
CATEGORICAL_FEATURES = ['surface_form_type']
CATEGORICAL_FEATURES_LIST = [
    'label_pref',
    'label_alt',
    'chunk_pref',
    'chunk_alt',
    'chunk_descr',
    'manual'
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = 'is_OK'

PARAM_GRID = {
    'classifier__n_estimators': [50, 100, 200, 400],
    'classifier__learning_rate': [0.01, 0.1, 0.5, 1],
    'classifier__max_depth': [1, 5, 10],
    'classifier__loss': ['ls'],
    'classifier__random_state': [9999],
}


# %%
# For prototyping
import seaborn as sns
from matplotlib import pyplot as plt
import altair as alt


# %% [markdown]
# ## 1. Create training data
#
# Import manual review tables and use them to create training data. The manual reviewers were performed by running earlier versions of the skills detection algorithm over a dataset of job adverts, and manually checking skills entities and surface forms that came up most frequently in the data.
#
# In this review, each surface form - skills entity pairs that was reviewed, was given an integer rating 1, 2 or 3:
# - 1=not good (remove the form)
# - 2=sub-optimal (try reassigning the form to a different entity)
# - 3=is ok

# %%
def get_manual_review_results(fpath = MANUAL_CHECK_DIR, exclude=[2]):
    """ Fetches results from manual review (exclude = [2] to remove ambiguous entries) """
    # First manual check
    checks_1 = pd.read_excel(fpath / 'v01_reed_extract_counts.xlsx')
    # Second manual check
    checks_2 = pd.read_excel(fpath / 'v01_r1_hackweek_extract_counts.xlsx')

    # Combine both checks
    columns = ['surface_form', 'surface_form_type', 'preferred_label', 'entity', 'manual_OK']
    checks = (checks_1[columns]
              .append(checks_2[columns], ignore_index=True)
              .drop_duplicates('surface_form', keep='last')
             )   
    # Exclude specific ratings
    checks = checks[
        (checks.manual_OK.isnull() == False)
        & (checks.manual_OK.isin(exclude) == False)]   
    return checks


# %% [markdown]
# ## 2. Calculate 'quality' measures

# %%
def create_skills_documents(
    fpath=DATA_PATH / 'interim/skills_docs.json',
    test=True,
    overwrite=False
):
    """ Create and preprocess skills 'documents' by joining up labels and the description """
    if os.path.exists(fpath) and not overwrite:
        with open(fpath, 'r') as file:
            skills_docs = json.load(file)
    else:
        # ESCO skills table
        skills = get_esco_skills()
        if test: skills = skills.iloc[0:5]
        # Create 'documents' by joining up labels and the description
        document_generator = skills_utils.create_documents(
            (
                skills.preferred_label.to_list(),
                [" ".join(split_string(s)) for s in skills.alt_labels.to_list()],
                skills.description.to_list(),
            )
        )
        # Preprocess the text
        skills_docs = {str(skills.iloc[i].id): clean_text(s) for i, s in enumerate(document_generator)}
        # Save the documents
        with open(fpath, 'w') as file:
            json.dump(skills_docs, file, indent=4)
    return skills_docs



# %%
def split_forms_entities(df):
    surface_forms = df['surface_form'].to_list()
    entities = df['entity'].to_list() 
    return surface_forms, entities

### TF-IDF based measures

def get_tfidf_token_indices(phrase, vocabulary):
    """ For a given phrase, get its tokens' indices in the tfidf matrix """
    # Get individual tokens
    tokens = phrase.split(" ")
    # Get token indices
    token_indexes = []
    for token in tokens:
        try:
            token_indexes.append(vocabulary[token])
        except KeyError:
            pass
    return token_indexes

def tfidf_phrase_sums(phrase, vocabulary, tfidf_matrix):
    """
    For the provided phrase, splits it into words (tokens) and, for each skills
    document, calculates the sum of the document's tf-idf vector elements that
    correspond to the words. This sum is assumed to indicate the phrase's
    representativeness of the skills document.
    """
    token_indexes = get_tfidf_token_indices(phrase, vocabulary)
    return (np.array(tfidf_matrix[:, token_indexes].sum(axis=1))
            .flatten()
            .tolist()
           )

def tfidf_phrase_means(phrase, vocabulary, tfidf_matrix):
    """
    """
    token_indexes = get_tfidf_token_indices(phrase, vocabulary)
    if len(token_indexes) > 0:
        return (np.array(tfidf_matrix[:, token_indexes].mean(axis=1))
                .flatten()
                .tolist()
               )
    else:
        return [0]*tfidf_matrix.shape[0]

    
def get_tfidf_scores(surface_form_df, vocabulary, tfidf_matrix, func):
    """
    Calculate tfidf term scores (either sums or means:
    pass func=tfidf_phrase_sums or func=tfidf_phrase_means respectively)
    """
    surface_forms, entities = split_forms_entities(surface_form_df)
    tfidf_entity_scores = []
    for i, phrase in enumerate(surface_forms):
        # Get the surface form tfidf scores (across all documents)
        tfidf_scores = func(phrase, vocabulary, tfidf_matrix)
        # Get the surface form's score corresponding to its entity
        score = tfidf_scores[entities[i]]
        # Collect the scores
        tfidf_entity_scores.append(score)
    return tfidf_entity_scores

### IDF based measures

def get_tfidf_vectorizer(documents):
    """
    Get a tfidf vectorizer 
    
    Args:
        documents (list of str): Documents to vectorise
    """
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1))
    tfidf_vectorizer.fit_transform(documents);
    return tfidf_vectorizer

def idf_phrase_values(phrase, tfidf_vectorizer):
    """ Get idf scores for each token in the phrase """
    token_indexes = get_tfidf_token_indices(phrase, tfidf_vectorizer.vocabulary_)
    phrase_idfs = tfidf_vectorizer.idf_[token_indexes]
    return phrase_idfs

def idf_phrase_sums(phrase, tfidf_vectorizer):
    """ Sum the idf scores of a phrase """
    return idf_phrase_values(phrase, tfidf_vectorizer).sum(axis=0)

def idf_phrase_means(phrase, tfidf_vectorizer):
    """ Average the idf scores of a phrase """
    idf = idf_phrase_values(phrase, tfidf_vectorizer)
    if len(idf) > 0:
        return idf.mean() 
    else:
        return 0

def get_idf_scores(surface_form_df, tfidf_vectorizer, func):
    """ """
    surface_forms, _ = split_forms_entities(surface_form_df)
    idf_scores = [func(sf, tfidf_vectorizer) for sf in surface_forms]
    return idf_scores

### Embedding-based measures

def get_best_embedding_similarities(surface_form_df, precomputed=None, bert_transformer=None):
    """ Compare with all labels, and choose the best match; precomputed='v01_1' """
    surface_forms, entities = split_forms_entities(surface_form_df)
    # Surface form embeddings
    if precomputed is None:
        emb_sf = np.array(bert_transformer.encode(surface_forms))
    else:
        emb_sf = np.load(DATA_PATH / f'processed/embeddings/surface_forms_{precomputed}_emb_{BERT_MODEL}.npy')
    # All label embeddings
    skill_ids, emb_labels = get_skill_embeddings(BERT_MODEL)
    # Calculate cosine similarity for each pair
    sims=[]
    for i in range(len(surface_form_df)):
        try:
            entity = entities[i]
            vector_indexes = np.where(skill_ids==entity)[0]
            sim = (1-cdist(emb_sf[i,:].reshape(1,-1), emb_labels[vector_indexes,:], metric='cosine')).max()
            sims.append(sim)  
        except:
            print(i)
        
    return sims   


# %%
def plot_dist(df, score):
    """ Plot scores for good (manual_OK=3) and bad (manual_OK=1) surface forms """
    df = df.copy()
    df['score'] = score
    sns.set_style("white")
    sns.displot(
        df[df.manual_OK.isin([1,3])], 
        x='score', 
        hue='manual_OK', 
        bins=25, 
        multiple="dodge",
        stat="density",
        palette=['r', 'b']
    )
    plt.show()


# %% [markdown]
# ## 3. Predictive model

# %%
def create_numerical_features(inputs, precomputed=None, bert_transformer=BERT_TRANSFORMER):
    """ Generate numerical features; 
    Example usage: precomputed = skills_model['surface_forms'], precomputed = skills_model['name'] """
    model_data = inputs.copy()
    
    # Embedding based features
    if 'score_emb_similarity' in NUMERICAL_FEATURES:
        model_data['score_emb_similarity'] = get_best_embedding_similarities(
            model_data, 
            precomputed=precomputed,
            bert_transformer=bert_transformer
        );
    
    # TF-IDF based features
    skills_docs = list(create_skills_documents().values())
    tfidf_matrix, vocabulary = tfidf_vectorise(skills_docs)
    
    if 'score_tfidf_sum' in NUMERICAL_FEATURES: 
        model_data['score_tfidf_sum'] = get_tfidf_scores(model_data, vocabulary, tfidf_matrix, tfidf_phrase_sums)
        
    if 'score_tfidf_mean' in NUMERICAL_FEATURES:         
        model_data['score_tfidf_mean'] = get_tfidf_scores(model_data, vocabulary, tfidf_matrix, tfidf_phrase_means) 
    
    # IDF based features
    tfidf_vectorizer = get_tfidf_vectorizer(skills_docs)
    
    if 'score_idf_sum' in NUMERICAL_FEATURES:
        model_data['score_idf_sum'] = get_idf_scores(model_data, tfidf_vectorizer, idf_phrase_sums)
    
    if 'score_idf_mean' in NUMERICAL_FEATURES:      
        model_data['score_idf_mean'] = get_idf_scores(model_data, tfidf_vectorizer, idf_phrase_means)
    
    return model_data


def generate_numerical_features_for_model_forms(
    skills_model,
    fpath=DATA_PATH/'processed/surface_forms',
    test=True,
    overwrite=False
):   
    """ """
    filepath = fpath / f"surface_forms_with_features_{skills_model['name']}.csv"
    if os.path.exists(filepath) and not overwrite:
        surface_forms_with_features = pd.read_csv(filepath)
    else:
        if not test:
            df = skills_model['surface_forms']
        else:
            df = skills_model['surface_forms'].sample(500).append(skills_model['surface_forms'].iloc[-50:], ignore_index=True)    
        surface_forms_with_features = create_numerical_features(df, precomputed=skills_model["name"])
        surface_forms_with_features.to_csv(filepath, index=False)
    return surface_forms_with_features


def create_prediction_target(model_data, target_column='is_OK'):
    """ Transform manual ratings between 0 and 1 """
    model_data=model_data.copy()
    model_data[target_column] = -1
    model_data.loc[(model_data.manual_OK==1), target_column] = 0
    model_data.loc[(model_data.manual_OK==3), target_column] = 1
    return model_data


def preprocessing_pipeline():
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())])
    categorical_transformer = OneHotEncoder(
        handle_unknown='ignore',
        categories=[CATEGORICAL_FEATURES_LIST]
    ) 
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)])
    return preprocessor


def classification_pipeline():
    return Pipeline(steps=[('classifier', GradientBoostingRegressor())])


def find_best_threshold(estimator, X, y,  make_plots=False): 
    """ Find the best threshold value for obtaining classification from model probabilities """
    Y_pred_prob = estimator.predict(X)
    threshold_values = np.linspace(0.1,0.9,50)
    F1_scores = []
    
    for t in threshold_values:
        Y_pred = (Y_pred_prob >= t).astype(int)
        F1_scores.append(f1_score(y, Y_pred, average="micro"))

    max_score = max(F1_scores)
    threshold_value = threshold_values[F1_scores.index(max_score)]
    
    if make_plots:
        F1_vs_threshold = pd.DataFrame({"threshold": threshold_values, "F1": F1_scores})
        plt.figure()
        ax = sns.lineplot(data=F1_vs_threshold, x="threshold", y="F1")
    
    return {'best_threshold': threshold_value, 'best_F1': max(F1_scores)}


def get_prediction_probability(inputs, preprocessor, clf):
    data = inputs.copy()
    data['predicted'] = clf.predict(preprocessor.transform(inputs[ALL_FEATURES]))
    return data


def get_prediction_discrete(inputs, t=0.55):
    data = inputs.copy()
    data['is_predicted_OK'] = (data['predicted'] >= t).astype(int)
    return data

def evaluate_performance(X, y, y_predict, t):
    """ X=Preprocessed data, y=ground truth labels, y_predict=predicted, t=threshold """
    is_OK = (y_predict > t).astype(int)
    print(classification_report(y, is_OK))

def create_test_param_grid(param_grid):
    test_param_grid = {}
    for key in param_grid:
        test_param_grid[key] = [param_grid[key][0]]
    return test_param_grid


# %%
def train_manual_review_rating_model(skills_model, test=False):
    """ Model that tries to predict ratings from the manual review """

    print("Preprocessing")
    # Get numerical features of the model surface forms
    model_data = generate_numerical_features_for_model_forms(skills_model, test=test)

    # Fit preprocessor on the full dataset from the skills model
    preprocessor = preprocessing_pipeline()

    # Data for training (from manual review)
    training_data = create_numerical_features(get_manual_review_results())
    training_data = create_prediction_target(training_data, TARGET_COLUMN)
    preprocessor.fit(training_data[ALL_FEATURES])

    # Prepare training dataset
    X = training_data[ALL_FEATURES]
    y = training_data[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=999, test_size=0.2, train_size=0.8)

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    X_transformed = preprocessor.transform(X)
    
    if test:
        param_grid = create_test_param_grid(PARAM_GRID)
    else:
        param_grid = PARAM_GRID

    print("Training the model") 
    # Train the model
    clf = classification_pipeline()
    grid_search = GridSearchCV(clf, param_grid, cv=10)
    grid_search.fit(X_train, y_train);
    clf.set_params(**grid_search.best_params_)
    clf.fit(X_train, y_train);   
    
    # Report evaluation
    print('----')
    print(f'Training set: (n={X_train.shape[0]})')
    thresh_train = find_best_threshold(clf, X_train, y_train)
    print(f"Best threshold is {thresh_train['best_threshold']:.2f} (F1={thresh_train['best_F1']: .2f})")
    evaluate_performance(X_train, y_train, clf.predict(X_train), thresh_train['best_threshold'])
    print(f'Test set: (n={X_test.shape[0]})')   
    thresh_test = find_best_threshold(clf, X_test, y_test)
    print(f"Best threshold is {thresh_test['best_threshold']:.2f} (F1={thresh_test['best_F1']: .2f})")    
    evaluate_performance(X_test, y_test, clf.predict(X_test), thresh_test['best_threshold'])
    print('----')
    
    thresh = find_best_threshold(clf, X_transformed, y)
    print('----')
    print(f'Evaluation on the combined training and test set: (n={X_transformed.shape[0]})')
    print(f"Best threshold is {thresh['best_threshold']:.2f} (F1={thresh['best_F1']: .2f})")
    evaluate_performance(X_transformed, y, clf.predict(X_transformed), thresh_train['best_threshold'])    
    print('----')    
    
    # Apply the model on the full dataset
    surface_forms_with_predictions = get_prediction_probability(model_data, preprocessor, clf)
    surface_forms_with_predictions = get_prediction_discrete(surface_forms_with_predictions, t=thresh['best_threshold'])  

    # Export the quality rating model
    quality_rating_model = {
        'preprocessor': preprocessor,
        'clf': clf,
        'best_params': grid_search.best_params_,
        'features': ALL_FEATURES,
        'best_threshold': thresh['best_threshold'],
        'best_F1': thresh['best_F1']
    }

    print("Updating the model surface forms table")  
    # Update the skills detection model
    new_model = skills_model.copy()
    new_model['surface_forms'] = (new_model['surface_forms']
                                  .merge(
                                      surface_forms_with_predictions[['entity', 'surface_form','predicted', 'is_predicted_OK']], 
                                      how='left', 
                                      on=['surface_form', 'entity'])
                                  .rename(columns={'predicted': 'predicted_q'})
                                 )
    new_model['name'] = f"{skills_model['name']}q"
    save_model_locally(new_model)
    save_model_in_s3(new_model)
    pickle_model(quality_rating_model, f"quality_rating_model_{new_model['name']}")

    return new_model, quality_rating_model


# %%
if __name__ == "__main__":
    new_model, quality_rating_model = train_manual_review_rating_model(skills_model, test=False)
    
    # Check ML model accuracy
    print('Check that this is the same as the report above:')
    training_data = create_numerical_features(get_manual_review_results())
    training_data = create_prediction_target(training_data, TARGET_COLUMN)
    df = get_prediction_discrete(
        get_prediction_probability(
            training_data,
            quality_rating_model['preprocessor'],
            quality_rating_model['clf']
        ), quality_rating_model['best_threshold']
    )
    print(classification_report(training_data[TARGET_COLUMN], df.is_predicted_OK.to_list()))
