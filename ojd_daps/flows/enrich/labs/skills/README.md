# Open Jobs Observatory: Skills extraction

## Welcome to skills lab!

Here, you'll find instructions for using the skills extraction algorithm and a description of its development.

In brief, the skills detection algorithm uses spacy [PhraseMatcher](https://spacy.io/api/phrasematcher), which looks for approximately 100,000 'surface forms', i.e. phrases that are linked to one of the 13,000+ [ESCO skills](https://ec.europa.eu/esco/portal/skill). These surface forms have been generated from ESCO skills labels and descriptions, and an attempt has been made to remove generic forms based on whether they are "representative" of their ESCO skills entities.

In addition, the skills have been grouped into categories using a consensus community detection algorithm. The final taxonomy can be located in the folder `taxonomy`.

With this code and the pickled model stored in the `models` folder, you can apply the skills detection algorithm on your text data (see the Usage section). However, because we are not able to share the raw input job advert text data, it will not be possible to perform most of the algorithm development steps outlined further below. Nonetheless, we hope that the code might serve as a template for further developments, and if you have access to a job advert text dataset you might be able to adjust the code to run on your data.

For a more detailed explanation of the project and the skills extraction algorithm you can check out [the project website]( https://www.nesta.org.uk/data-visualisation-and-interactive/open-jobs-observatory/) and [blogs](https://www.nesta.org.uk/project-updates/skills-extraction-OJO/).

## Installation

In addition to the main ojd_daps requirements, you will need to install additional packages specified in the skills lab `requirements.txt`. The best approach is probably to clone your ojd_daps environment and add the additional packages there.

```
$ conda create --clone <your_ojd_daps_environment> --name ojd_daps_skills_labs
$ conda activate ojd_daps_skills_labs
$ pip install -r requirements.txt
```

## Usage

### Using the skills detection algorithm

You can detect skills in a text in the following way:

```python
from ojd_daps.flows.enrich.labs.skills.skills_detection_utils import (
  load_model,
  setup_spacy_model,
  detect_skills,
  clean_text)

model = load_model("v02_1", from_local=True)
nlp = setup_spacy_model(model["nlp"])
detect_skills(clean_text(job_description_text), model, nlp)
```

### Examples

To try out the algorithm, run the following lines. Note that you can set `return_dict=False` to output a pandas dataframe instead.

```python
# Agile business analyst
text = 'Support the Business to continually improve the business needs by working hand-in-hand with the Product Owner and Development teams. Work with product owners to write & develop clear, non-implementation specific epics, user stories and acceptance criteria. Interview product owners to understand as-is business processes and then develop customer-driven to-be processes. Progressively improve our demand side and analysis practices, focusing on waste elimination, demonstrating this improvement with hard data. Enforce and promote SCRUM disciplines. Should be able to organize and run daily stand ups. Promotes healthy team environment and removes impediments Ensures team is delivering / aligned on project vision and goals A strong background in retail financial services and a proven ability to quickly understand the business strategy and objectives Significant experience of business analysis in an agile environment An IT background with solid grounding in technology is essential Experience of developing high quality user stories and acceptance criteria for multiple business processes having multiple product owners in an organization new to agile development methodology 3+ years in the role of Business Analyst Communication, group dynamics, collaboration and continuous improvement are - core being best practice driven Kanban practitioner, Scrum Certified or Six Sigma certification a plus 1 or higher in a related discipline from an academic institution; Masters a plus'
detect_skills(clean_text(text), model, nlp, return_dict=True)

[{'surface_form': 'team work',
  'surface_form_type': 'manual',
  'preferred_label': 'work as a team',
  'entity': 10228,
  'predicted_q': 0.8562354219165966,
  'cluster_0': 0.0,
  'cluster_1': 0.0,
  'cluster_2': 0.0,
  'label_cluster_0': 'Transversal skills',
  'label_cluster_1': 'General Workplace Skills',
  'label_cluster_2': 'General Workplace Skills'},
 {'surface_form': 'communication',
  'surface_form_type': 'label_pref',
  'preferred_label': 'communication',
  'entity': 1139,
  'predicted_q': 0.4968150938837877,
  'cluster_0': 0.0,
  'cluster_1': 0.0,
  'cluster_2': 0.0,
  'label_cluster_0': 'Transversal skills',
  'label_cluster_1': 'General Workplace Skills',
  'label_cluster_2': 'General Workplace Skills'},
 {'surface_form': 'business analysis',
  'surface_form_type': 'label_pref',
  'preferred_label': 'business analysis',
  'entity': 5288,
  'predicted_q': 0.9144979542147392,
  'cluster_0': 5.0,
  'cluster_1': 10.0,
  'cluster_2': 24.0,
  'label_cluster_0': 'Business Administration, Finance & Law',
  'label_cluster_1': 'Business Administration',
  'label_cluster_2': 'Business & Project Management'},
 {'surface_form': 'agile development',
  'surface_form_type': 'label_pref',
  'preferred_label': 'Agile development',
  'entity': 11555,
  'predicted_q': 0.919138661554582,
  'cluster_0': 5.0,
  'cluster_1': 10.0,
  'cluster_2': 24.0,
  'label_cluster_0': 'Business Administration, Finance & Law',
  'label_cluster_1': 'Business Administration',
  'label_cluster_2': 'Business & Project Management'},
 {'surface_form': 'business process',
  'surface_form_type': 'label_pref',
  'preferred_label': 'business processes',
  'entity': 8412,
  'predicted_q': 0.9260316554658904,
  'cluster_0': 5.0,
  'cluster_1': 10.0,
  'cluster_2': 24.0,
  'label_cluster_0': 'Business Administration, Finance & Law',
  'label_cluster_1': 'Business Administration',
  'label_cluster_2': 'Business & Project Management'},
 {'surface_form': 'agile',
  'surface_form_type': 'label_alt',
  'preferred_label': 'ICT project management methodologies',
  'entity': 10023,
  'predicted_q': 0.7973309852784515,
  'cluster_0': 5.0,
  'cluster_1': 10.0,
  'cluster_2': 24.0,
  'label_cluster_0': 'Business Administration, Finance & Law',
  'label_cluster_1': 'Business Administration',
  'label_cluster_2': 'Business & Project Management'},
 {'surface_form': 'scrum',
  'surface_form_type': 'label_alt',
  'preferred_label': 'ICT project management methodologies',
  'entity': 10023,
  'predicted_q': 0.8735558706143535,
  'cluster_0': 5.0,
  'cluster_1': 10.0,
  'cluster_2': 24.0,
  'label_cluster_0': 'Business Administration, Finance & Law',
  'label_cluster_1': 'Business Administration',
  'label_cluster_2': 'Business & Project Management'},
 {'surface_form': 'continuous improvement',
  'surface_form_type': 'chunk_pref',
  'preferred_label': 'create a work atmosphere of continuous improvement',
  'entity': 8367,
  'predicted_q': 0.6559178410382149,
  'cluster_0': 5.0,
  'cluster_1': 10.0,
  'cluster_2': 24.0,
  'label_cluster_0': 'Business Administration, Finance & Law',
  'label_cluster_1': 'Business Administration',
  'label_cluster_2': 'Business & Project Management'},
 {'surface_form': 'financial service',
  'surface_form_type': 'chunk_pref',
  'preferred_label': 'offer financial services',
  'entity': 11948,
  'predicted_q': 0.7445369673332084,
  'cluster_0': 5.0,
  'cluster_1': 11.0,
  'cluster_2': 27.0,
  'label_cluster_0': 'Business Administration, Finance & Law',
  'label_cluster_1': 'Finance & Law',
  'label_cluster_2': 'Financial Services'},
 {'surface_form': 'kanban',
  'surface_form_type': 'chunk_descr',
  'preferred_label': 'continuous improvement philosophies',
  'entity': 236,
  'predicted_q': 0.1346155465336057,
  'cluster_0': 0.0,
  'cluster_1': 0.0,
  'cluster_2': 0.0,
  'label_cluster_0': 'Transversal skills',
  'label_cluster_1': 'General Workplace Skills',
  'label_cluster_2': 'General Workplace Skills'}]
```

```python
# Counterbalance forklift driver
text = 'Counterbalance Forklift Driver, £9.20per hour, temp to perm position Monday to Friday, 6am to 5pm, 50 hours a week  Company in Brandon, Suffolk Main Duties:  Make sure you follow the health and safety rules on siteStock controlMove stock using your forkliftGeneral warehouse duties Essential Skills:  Previous experience in a similar role would be advantageMust have a valid (ITSAAR or RTITB)  forklift counterbalance license If you would like to apply for this role, email your current up to date CV to with a covering letter outlining your suitability for the role, ensure that your CV includes examples of the above as applicable Unfortunately, due to the large volume of applications we receive, we can only respond to applicants with relevant work experience. All applicants must live and be eligible to work in the UK. Hales Group Ltd collects and keeps information from applicants, so that we can monitor our recruitment process, ensure compliance with the Equal Opportunities policy, and when appropriate send you details of future job opportunities. We keep your name and  address, and details of your application. If you do not want us to do this please contact your local branch.'
detect_skills(clean_text(text), model, nlp, return_dict=True)

[{'surface_form': 'warehouse',
  'surface_form_type': 'label_alt',
  'preferred_label': 'perform warehousing operations',
  'entity': 6544,
  'predicted_q': 0.7634879365328688,
  'cluster_0': 6.0,
  'cluster_1': 12.0,
  'cluster_2': 31.0,
  'label_cluster_0': 'Engineering, Construction & Maintenance',
  'label_cluster_1': 'Manufacturing & Engineering',
  'label_cluster_2': 'Manufacturing & Mechanical Engineering'},
 {'surface_form': 'move stock',
  'surface_form_type': 'label_alt',
  'preferred_label': 'transfer stock',
  'entity': 13252,
  'predicted_q': 0.91557337248465,
  'cluster_0': 6.0,
  'cluster_1': 12.0,
  'cluster_2': 31.0,
  'label_cluster_0': 'Engineering, Construction & Maintenance',
  'label_cluster_1': 'Manufacturing & Engineering',
  'label_cluster_2': 'Manufacturing & Mechanical Engineering'},
 {'surface_form': 'health safety',
  'surface_form_type': 'label_alt',
  'preferred_label': 'health and safety in the workplace',
  'entity': 1335,
  'predicted_q': 0.934989083752494,
  'cluster_0': 6.0,
  'cluster_1': 13.0,
  'cluster_2': 34.0,
  'label_cluster_0': 'Engineering, Construction & Maintenance',
  'label_cluster_1': 'Construction, Installation & Maintenance',
  'label_cluster_2': 'Workplace Safety Management'},
 {'surface_form': 'forklift',
  'surface_form_type': 'chunk_pref',
  'preferred_label': 'operate forklift',
  'entity': 2185,
  'predicted_q': 0.4996100526210357,
  'cluster_0': 6.0,
  'cluster_1': 12.0,
  'cluster_2': 31.0,
  'label_cluster_0': 'Engineering, Construction & Maintenance',
  'label_cluster_1': 'Manufacturing & Engineering',
  'label_cluster_2': 'Manufacturing & Mechanical Engineering'},
 {'surface_form': 'health safety rule',
  'surface_form_type': 'chunk_pref',
  'preferred_label': 'foster compliance with health and safety rules by setting an example',
  'entity': 11995,
  'predicted_q': 0.8894357124558738,
  'cluster_0': 6.0,
  'cluster_1': 13.0,
  'cluster_2': 34.0,
  'label_cluster_0': 'Engineering, Construction & Maintenance',
  'label_cluster_1': 'Construction, Installation & Maintenance',
  'label_cluster_2': 'Workplace Safety Management'},
 {'surface_form': 'safety rule',
  'surface_form_type': 'chunk_alt',
  'preferred_label': 'work with respect for own safety',
  'entity': 9662,
  'predicted_q': 0.6125117663771006,
  'cluster_0': 6.0,
  'cluster_1': 13.0,
  'cluster_2': 34.0,
  'label_cluster_0': 'Engineering, Construction & Maintenance',
  'label_cluster_1': 'Construction, Installation & Maintenance',
  'label_cluster_2': 'Workplace Safety Management'}]
```

Notes:
- `entity` is the ESCO skill identifier number corresponding to `id` column in `data/raw/esco/ESCO_skills_hierarchy.csv`
- `predicted_q` is the "quality" of the surface form predicted by a trained machine learning model (i.e. how representative it is of the skill entity; higher is better). Normally, only the surface forms above a learned threshold are shown (unless you define `debug=True`)
- `cluster_{x}` provides the cluster integer label. Note that some surface forms have not been assigned to a cluster and hence have `null` values. These will not be displayed in the output unless you set `debug=True`.
- All surface forms (both above and below the "quality" indicator threshold) can be accessed via `model['surface_forms']`

## Algorithm development

The following section describes the flows and scripts that were developed to build the skills extraction algorithm. Because we are not able to share the raw input job advert text data, it will not be possible to perform most of the steps. Nonetheless, the code might serve as a template for further developments, and if you have access to a job advert text dataset you might be able to adjust the code to run on your data.

### Skills lab folder structure

Here is the suggested way of setting up the folder structure for this project when developing the skills extraction algorithm.

```
  ├── data
  │   ├── raw                   <- The original, immutable data
  │   ├── interim               <- Intermediate output data that has been transformed
  │   ├── aux                   <- Manually created data (e.g. manual review results)
  │   ├── processed             <- Final, processed data sets
  │   │   ├── detected_skills   <- Stores outputs from batch_detection_flow.py and PIPELINE_detect_surface_forms.py
  │   │   ├── embeddings        <- Sentence and word2vec/skill2vec embeddings of skills labels and surface forms
  │   │   ├── surface_forms     <- Outputs related to surface form generation
  │   │   ├── clusters          <- Outputs related to cluster generation  
  │── models                    <- Models used for skills extraction, and tables with their surface forms
  │── notebooks                 <- Prototyping notebooks with elements of the model building pipeline that
  │                                have not been factored out into metaflow flows
  │── tests                     <- Tests for some of the utils
```

### Running flows & scripts

The algorithm was built by running the following commands, in the order as shown below. One can also run all of the steps with one command, using the shell script `make_model_v02.sh`. Ideally, these steps should be refactored as neater metaflow flows.

#### Step 1: Flow to generate surface forms

Flow that generates surface forms from ESCO skills labels and descriptions, and checks their representativeness.  Note that any surface forms that have been discarded in this process are recorded in the file `data/processed/surface_forms/surface_forms_removed_<model_name>.json` for later inspection.
```
$ python surface_forms_flow.py run --production True --model_name v01_1
```

#### Step 2: Flow to detect surface forms in test data

Flow that detects skills in a batch of job advertisement texts and stores the results in a file
`data/processed/detected_skills/<dataset_name>_<model_name>_skills.json`. The default
  dataset used in this step was a small test dataset with adverts collected across a few days `data/raw/job_ads/sample_reed_extract.csv`.
```
$ python batch_detection_flow.py run --production True --model_name v01_1
```

#### Step 3: Remove generic surface forms

Flow that refines the model by removing the most frequently occurring surface forms (based on the small test dataset) while keeping certain, manually selected forms.
```
$ python remove_frequent_forms_flow.py run --detected_skills_path data/processed/detected_skills/sample_reed_extract_v01_1_skills.json --new_model_name v01_1
```

#### Step 4: Estimating surface form quality

Python script that builds a model to predict surface form quality.
```
$ python notebooks/PIPELINE_assess_quality.py
```
NB: There is an intermediate step (between Step 3 and 4) to precompute sentence embeddings of skills labels and all surface forms, and place them in `data/processed/embeddings`. For this, I used a [Google Colab notebook](https://colab.research.google.com/drive/1kH-HbhxK4VAL7-h4deCW_cIf90OzAHEQ?usp=sharing) as a very simple solution to calculate 100k+ embeddings using a GPU (you could also use, e.g. an AWS instance). Note that if you change anything in Steps 1-3, you might need to re-calculate the surface form embeddings and place the new ones in the aforementioned folder.

#### Step 5: Defining transversal skills

Python script to define two clusters of general workplace skills and language skills (these won't be included in the clustering step).

NB: There was an intermediate step before Step 5, where surface forms were detected in a larger collected dataset (spanning between Dec 2020 and mid-May 2021). The frequency and co-occurrence statistics of surface forms are used in a few further steps to help define transversal skills and develop data-driven skills taxonomy. The skills are detected from a database dump of jobs ads, stored in `data/raw/job_ads` using `notebooks/PIPELINE_detect_surface_forms.py`. The detection, without any optimisation and parallelisation, takes about 16 hrs for 1M job ads.

```
$ python notebooks/PIPELINE_general_skills.py
```

#### Step 6: Clustering surface forms

Python script to cluster surface forms using consensus community detection, into three hierarchical levels.
```
$ python notebooks/PIPELINE_surface_form_clustering.py
```

#### Step 7: Assigning skills to surface form clusters

Using surface form clustering results, to assign skills to clusters.
```
$ python notebooks/PIPELINE_cluster_assignments.py
```

#### Step 8: Manually reviewing cluster names and assignments

Manual adjustments and further refinements to produce the final skills taxonomy. See `notebooks/AUX_review_clusters.py` for details.


## Skills algorithm version history
- `v01` (April 2021) - First version (now defunct)
- `v01_1` (June 2021) - Refactored the code, and slightly changed the api of the `detect_skills` function (defunct)
- `v02` (June 2021) - Outputs cluster integer labels (defunct)
- `v02_1` (July 2021) - Reviewed surface form cluster assignments, cluster hierarchy, and provided cluster text labels
