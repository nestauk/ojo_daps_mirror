# Open Jobs Observatory: Skills extraction

## Welcome to skills lab!

Here, you'll find the development of the skills extraction algorithm.

In brief the skills detection algorithm uses spacy [PhraseMatcher](https://spacy.io/api/phrasematcher), which looks for approximately 100,000 'surface forms', i.e. phrases that are linked to one of the 13,000+ [ESCO skills](https://ec.europa.eu/esco/portal/skill). These surface forms have been generated from ESCO skills labels and descriptions, and an attempt has been made to remove generic forms based on (i) whether they are "representative" of their ESCO skills entities, and (ii) their frequency in the data (forms that are very frequent have been removed).

In addition, the skills have been grouped into categories using a consensus community detection algorithm.

For a more detailed explanation of the problem and the approach taken so far, you can check out [these (a bit outdated) slides](https://docs.google.com/presentation/d/1dwyrYraKoVGY3admtxZrFDG7kIPGZBiIgfZxdjIKHlA/edit#slide=id.g5756df2193_0_64).

## Skills lab folder structure

```
  ├── data
  │   ├── raw                   <- The original, immutable data
  │   ├── interim               <- Intermediate output data that has been transformed
  │   ├── aux                   <- Manually created data (mostly pertaining to manual labelling of skills)
  │   ├── processed             <- Final, processed data sets
  │   │   ├── detected_skills   <- Stores outputs from batch_detection_flow.py and PIPELINE_detect_surface_forms.py
  │   │   ├── embeddings        <- Sentence and word2vec embeddings of skills labels and surface forms
  │   │   ├── surface_forms     <- Outputs related to surface form generation
  │   │   ├── clusters          <- Outputs related to cluster generation  
  │── models                    <- Models used for skills extraction, and tables with their surface forms
  │── notebooks                 <- Prototyping notebooks with elements of the model building pipeline that 
  │                                have not yet been factored out into flows
  │── tests                     <- Tests for some of the utils 
```

## Installation

### Additional packages
In addition to the ojd_daps requirements, you will need to install packages specified in the skills lab `requirements.txt` as well. The best approach is probably to clone your ojd_daps environment and add the additional packages there.

```
$ conda create --clone <your_ojd_daps_environment> --name ojd_daps_skills_labs
$ conda activate ojd_daps_skills_labs
$ pip install -r requirements.txt
```

### Download input data

It should be possible to detect skills without manually downloading additional data - the specified model will be automatically fetched from s3.
To run all of the flows, however, you'll have to download the input data. This can be easily done by running the following command in the terminal, which will synchronise your local `data` (and `models`) folders with the copy on our S3 cloud storage

```
make sync_data_from_s3
```

## Usage

### Using the skills detection algorithm

Presently you can detect skills in a text in the following way

```python
from ojd_daps.flows.enrich.labs.skills.skills_detection_utils import (
  load_model,
  setup_spacy_model,
  detect_skills,
  clean_text)

model = load_model("v02_1")
nlp = setup_spacy_model(model["nlp"])
detect_skills(clean_text(job_description_text), model, nlp)
```

### Running flows & scripts

For building the model, you will need to run the following commands, in the order as shown below (it might take around 30+ mins).

You can also try running all of the steps with one command using the shell script `make_model_v02.sh` (I will eventually need to refactor this into a proper flow).

#### Step 1: Flow to generate surface forms

Flow that generates surface forms from ESCO skills labels and descriptions, and checks their representativeness (see [slide 18](https://docs.google.com/presentation/d/1dwyrYraKoVGY3admtxZrFDG7kIPGZBiIgfZxdjIKHlA/edit#slide=id.gcebb639646_0_1468) for more details on the latter). Note that any surface forms that have been discarded in this process are recorded in the file `data/processed/surface_forms/surface_forms_removed_<model_name>.json` for later inspection.
```
$ python surface_forms_flow.py run --production True --model_name v01_1
```

#### Step 2: Flow to detect surface forms in test data

Flow that detects skills in a batch of job advertisement texts and stores the results in a file
`data/processed/detected_skills/<dataset_name>_<model_name>_skills.json`. The default
  dataset is the test sample from Reed in `data/raw/job_ads/sample_reed_extract.csv`.
```
$ python batch_detection_flow.py run --production True --model_name v01_1
```

#### Step 3: Remove generic surface forms

Flow that refines the model by removing the most frequently occurring surface forms (based on the smaller test dataset)
```
$ python remove_frequent_forms_flow.py run --detected_skills_path data/processed/detected_skills/sample_reed_extract_v01_1_skills.json --new_model_name v01_1
```

#### Step 4: Estimating surface form quality

Python script that builds a model to predict surface form quality
```
$ python notebooks/PIPELINE_assess_quality.py
```
NB: There is an intermediate step (between Step 3 and 4) to precompute sentence embeddings of skills labels and all surface forms, and place them in `data/processed/embeddings`. For this, I used a [Google Colab notebook](https://colab.research.google.com/drive/1kH-HbhxK4VAL7-h4deCW_cIf90OzAHEQ?usp=sharing) as calculating 100k+ embeddings is much faster when using a GPU. If you want to repeat the analysis, you can just download the embeddings when doing `make sync_with_s3`. However, note that if you change anything in Steps 1-3, you might need to re-calculate the _surface form_ embeddings and place the new ones in the aforementioned folder. So, this is very hacky at the moment, and I should make this step also a part of the pipeline!

#### Step 5: Defining general skills

Python script to define two clusters of general and language skills (these won't be included in the clustering step).

NB: There is another intermediate step (For Steps 5+ onwards) to detect surface forms in the large dataset (Dec 2020 - May 2021), because the frequency and co-occurrence statistics of surface forms will be used in a few further steps. The skills are detected from a database dump of jobs ads, stored in `data/raw/job_ads` using `notebooks/PIPELINE_detect_surface_forms.py`. The detection, without any optimisation and parallelisation, takes about 16 hrs (for 1M job ads). Now, if you don't change anything in the Steps 1-3, you can already use the completed results from `data/processed/detected_skills`.

```
$ python notebooks/PIPELINE_general_skills.py
```

#### Step 6: Clustering surface forms

Python script to cluster surface forms
```
$ python notebooks/PIPELINE_surface_form_clustering.py
```
#### Step 7: Assigning skills to surface form clusters

Using surface form clustering results, to assign skills to clusters
```
$ python notebooks/PIPELINE_cluster_assignments.py
```

#### Step 8: Reviewing cluster names and assignments

Check `notebooks/AUX_review_clusters.py` for details.

## Examples

To replicate the successes and failures shown in [slides 19-21](https://docs.google.com/presentation/d/1dwyrYraKoVGY3admtxZrFDG7kIPGZBiIgfZxdjIKHlA/edit#slide=id.gcebb639646_0_1526), load in the model `'v01_1'` and run the following lines (note that there might be small differences from the slides).

```python
import pandas as pd
job_ads = pd.read_csv('data/raw/job_ads/sample_reed_extract.csv')
```

```python
# Software developer
detect_skills(clean_text(job_ads.iloc[107].description), model, nlp, return_dict=True)

[{'surface_form': 'java script',
  'preferred_label': 'JavaScript',
  'entity': 3249,
  'surface_form_type': 'label_pref'},
 {'surface_form': 'c#',
  'preferred_label': 'C#',
  'entity': 4011,
  'surface_form_type': 'label_pref'},
 {'surface_form': 'sql',
  'preferred_label': 'SQL',
  'entity': 4761,
  'surface_form_type': 'label_pref'},
 {'surface_form': 'asp net',
  'preferred_label': 'ASP.NET',
  'entity': 4616,
  'surface_form_type': 'label_pref'},
 {'surface_form': 'java',
  'preferred_label': 'Java (computer programming)',
  'entity': 1340,
  'surface_form_type': 'chunk_pref'},
 {'surface_form': 'html',
  'preferred_label': 'use markup languages',
  'entity': 553,
  'surface_form_type': 'chunk_alt'}]
```

```python
# Forklift driver
detect_skills(clean_text(job_ads.iloc[5].description), model, nlp, return_dict=True)


[{'surface_form': 'move stock',
  'preferred_label': 'transfer stock',
  'entity': 13252,
  'surface_form_type': 'label_alt'},
 {'surface_form': 'job opportunity',
  'preferred_label': 'job market offers',
  'entity': 3035,
  'surface_form_type': 'label_alt'},
 {'surface_form': 'health safety',
  'preferred_label': 'health and safety in the workplace',
  'entity': 1335,
  'surface_form_type': 'label_alt'},
 {'surface_form': 'warehouse',
  'preferred_label': 'perform warehousing operations',
  'entity': 6544,
  'surface_form_type': 'label_alt'},
 {'surface_form': 'health safety rule',
  'preferred_label': 'foster compliance with health and safety rules by setting an example',
  'entity': 11995,
  'surface_form_type': 'chunk_pref'},
 {'surface_form': 'forklift',
  'preferred_label': 'operate forklift',
  'entity': 2185,
  'surface_form_type': 'chunk_pref'},
 {'surface_form': 'safety rule',
  'preferred_label': 'work with respect for own safety',
  'entity': 9662,
  'surface_form_type': 'chunk_alt'},
 {'surface_form': 'equal opportunity policy',
  'preferred_label': 'set inclusion policies',
  'entity': 6527,
  'surface_form_type': 'chunk_alt'}]
```

```python
# Agile business analyst
detect_skills(clean_text(job_ads.iloc[-1].description), model, nlp, return_dict=True)


[{'surface_form': 'business process',
  'preferred_label': 'business processes',
  'entity': 8412,
  'surface_form_type': 'label_pref'},
 {'surface_form': 'agile development',
  'preferred_label': 'Agile development',
  'entity': 11555,
  'surface_form_type': 'label_pref'},
 {'surface_form': 'communication',
  'preferred_label': 'communication',
  'entity': 1139,
  'surface_form_type': 'label_pref'},
 {'surface_form': 'business analysis',
  'preferred_label': 'business analysis',
  'entity': 5288,
  'surface_form_type': 'label_pref'},
 {'surface_form': 'scrum',
  'preferred_label': 'ICT project management methodologies',
  'entity': 10023,
  'surface_form_type': 'label_alt'},
 {'surface_form': 'agile',
  'preferred_label': 'ICT project management methodologies',
  'entity': 10023,
  'surface_form_type': 'label_alt'},
 {'surface_form': 'continuous improvement',
  'preferred_label': 'create a work atmosphere of continuous improvement',
  'entity': 8367,
  'surface_form_type': 'chunk_pref'},
 {'surface_form': 'financial service',
  'preferred_label': 'offer financial services',
  'entity': 11948,
  'surface_form_type': 'chunk_pref'},
 {'surface_form': 'kanban',
  'preferred_label': 'continuous improvement philosophies',
  'entity': 236,
  'surface_form_type': 'chunk_descr'},
 {'surface_form': 'grounding',
  'preferred_label': 'perform small vessel navigation',
  'entity': 4777,
  'surface_form_type': 'chunk_descr'},
 {'surface_form': 'team work',
  'preferred_label': 'manage work',
  'entity': 4614,
  'surface_form_type': 'chunk_alt'}]
```


## Skills algorithm version history
- `v01` (April 2021) - First version, now defunct
- `v01_1` (June 2021) - Refactored the code, and slightly changed the api of the `detect_skills` function
- `v02` (June 2021) - Outputs cluster integer labels
- `v02_1` (July 2021) - Reviewed surface form cluster assignments, cluster hierarchy, and provided cluster text labels
