# Data Quality Analysis

**Before contributing, please read the guidance on the front page of this repository.**

- [Getting Data](#getting-data)
  - [Collect: Raw HTML (S3)](#collect-raw-html-s3)
  - [Extract: Raw Job Ads (DB)](#extract-raw-job-ads-db)
  - [Enrich: Job Ad Locations (DB)](#enrich-job-ad-locations-db)
  - [Enrich: Job Ad Salaries (DB)](#enrich-job-ad-salaries-db)
  - [Enrich: Job Ad Vectors (DB)](#enrich-job-ad-description-vectors-db)
- [Contributing a Data Quality Analysis](#contributing-a-data-quality-analysis)

The purpose of this submodule is to find issues and quirks in the data quality and also in the data collection process. The output of the DQA should always be a "plotting module" (as defined below, but also see examples in the `plots` directory), which is required to have a standard form. In the future these plots can then be run to create a standard monitoring dashboard for OJD DAPS. Even if the plots aren't objectively interesting, they will form the basis of our understanding of the robustness of Open Jobs!

On top of the standard requirements, you will also need to run (first time, in your activated python environment):

```bash
pip install -r requirements.txt
```

from this directory (`ojd_daps/ojd_daps/dqa`), and also do (every time):

```bash
export PYTHONPATH=$PWD
```

from the project root directory (`ojd_daps`).

## Getting job ad data quickly

The quickest way to get job ad data is by using predefined 6-week "stock" windows and (automatically) downloading the query cache.

To do this, you will need to (from this submodule `ojd_daps/dqa`) do:

```python
from data_getters import get_cached_job_ads  # will automatically download the cache, if it is out-of-date

job_ads = get_cached_job_ads('21-12-2020', '01-02-2021')
```

A list of the permitted "stock" windows can be found using:

```python
from data_getters import get_valid_cache_dates

dates = get_valid_cache_dates()

dates[0]

>>> ('21-12-2020', '01-02-2021')
```

Note that for the above processes you _don't_ need to be signed into the Nesta VPN (or be sat in Nesta HQ), however if you would like to retrieve job descriptions you will need to sign into the VPN prior to fetching data from the database:

```python
from data_getters import fetch_descriptions

fetch_descriptions(['42640775', '42640774'])

>>> {'42640774': '[ Mu client and accountancy practice is looking for an experienced Bookkeeper to assist in managing their day-to-day accounting and finance requirements. Confidentiality, excellent organisational skills and accuracy are important qualifications for this position,  as well as good customer relations and the ability to communicate clearly. The ideal candidate for this position is a skilled multi-tasker, is reliable and is committed to consistently meeting deadlines. Duties:  Balance and maintain accurate ledgersMatch purchase orders with invoicesCoordinate bank deposits and report financial results on a regular basis to managementMonitor office expenses and tally and enter cash receiptsPay vendor invoices and track bank account balancesDevelop monthly financial statements, including cash flow, profit and loss statements and balance sheetsPrepare quarterly and monthly tax returns, along with payroll, operating and business taxes If the above is of interest to you then please APPLY NOW! ]',
 '42640775': "[ We have a fantastic opportunity for an Apprentice seeking a role which will allow them to learn on the job, work with a fun and enthusiastic team in a beautiful setting and work their way up in the Education sector. Starting apprentice wage of £4.20per hour - then increase wages based on review of work performance 35 hours a week 8-4pm (this time is not rigid) , . Immediate start. The job role will include:  Answering telephone queriesBooking appointmentsPhotocopying and printing documentsPosting letters / opening letters Stock taking of resources/ equipmentDrafting staff weekly rotaUpdating children's time tables / register, mealsRaising invoices and sending out invoicesAttending staff meetings / note takingCovering in the rooms when neededAny other duties to promote the smooth running of the business Personal Specification:  Must be keen to learn,Attention to detail is importantGood communicationPositive attitude to workCustomer service focussed ]"}
```

## Getting raw data from source (database or S3)

The `data_getters` submodule in this directory will help you to get ahold of both raw and processed data. The code is well documented, and if you are curious about functionality we recommend that you inspect the docstrings.

NB: You won't be able to retrieve data from the DB without turning on your VPN or being sat in Nesta HQ.

### [Collect] Raw HTML (S3)

Get the Raw HTML for job adverts. As per the corresponding DQA, this takes ages if `read_body=True` and so take a random subsample using `sample_ratio < 1` (normally around `0.01`). Setting `read_body=False` drops the HTML from the response and so speeds up the process, if all you need is metadata.

```python
from ojd_daps.dqa.data_getters import get_s3_job_ads

job_ads = get_s3_job_ads(job_board='reed', read_body=True, sample_ratio=0.01)
```

### [Extract] Raw Job Ads (DB)

```python
from ojd_daps.dqa.data_getters import get_db_job_ads

job_ads = get_db_job_ads()
```

If you need fewer job adverts, use `limit = 10000`, etc.

If you would also like features (locations, salaries, soc, etc) then use `return_features=True`. Features are returned as follows (for a single row of data):

```yaml
{
  "__version__": "21.06.23.187_extract",
  "id": "41547513",
  "data_source": "Reed",
  "created": datetime.datetime(2020,
  12,
  11,
  0,
  0),
  "url": None,
  "s3_location": "reed-41547513_test-False.txt",
  "job_title_raw": "Account Manager - Oxford",
  "job_location_raw": "Grove, Oxfordshire",
  "job_salary_raw": "65000.0000-65000.0000",
  "company_raw": "Executive Network Group",
  "contract_type_raw": "Permanent",
  "closing_date_raw": None,
  "description": "[]",
  # Note: features here, nested under the 'features' key
  "features": {
      "salary":
        {
          "__version__": "21.07.08.196_enrich",
          "min_salary": 65000.0,
          "max_salary": 65000.0,
          "rate": "per annum",
          "min_annualised_salary": 65000.0,
          "max_annualised_salary": 65000.0,
        },
      ? "location" # salaries
      : { "nuts_2_code": "UKI6", "nuts_2_name": "Outer London — South" },
      "soc": { "soc_code": "3534", "soc_title": "Manager, account" },
    }, # locations # soc # features
} # job_ad
```

Notes:

- if a feature could not be predicted for a given job ad then it will be omitted from the `features` collection.
- There is an initial one-off cost for collecting all of the features together (via `get_features` nested with `get_db_job_ads`) that will happen for the very first job ad, and thereafter the time will scale ~linearly with the number of job ads - so expect not too much difference between 10 ads and 1000 ads, though it should take a while above 10k ads as you're then limited by the time taken to retrieve the raw ads, rather than the features.

### [Enrich] Job Ad Locations (DB)

Get a lookup of Job Ad ID to a standardised location. Choose a location type from one of:

```
lad_18
health_12
region_18
nuts_0
nuts_1
nuts_2
nuts_3
```

and then do:

```python
from ojd_daps.dqa.data_getters import get_locations
job_locations = list(get_locations('reed', level='nuts_2'))
```

The output will be a deduplicated list of Job Ad IDs and the Location code, for example:

```python
{'job_id': '41600360', 'nuts_2_code': 'UKE3'}
```

In order to get names for each location, you should retrieve the location metadata using:

(NB: the warning is expected, this is due to the underlying ONS metadata which is a bit pants)

```python
from ojd_daps.dqa.data_getters import get_location_lookup
lookup = get_location_lookup()

lookup["UKE3"]
>>> "South Yorkshire"
```

### [Enrich] Job Ad Salaries (DB)

Get a lookup of Job Ad ID to a standardised salary, which is also normalised to "per annum":

```python
from ojd_daps.dqa.data_getters import get_salaries
salaries = get_salaries()

salaries[0]
>>> {'job_id': 41547514,
	 'normalised_salary': 55255.200000000004,
	 'salary': 212.52,
	 'rate': 'per day'}
```

### [Enrich] Job Ad description vectors (DB)

BERT vectorised job advert descriptions. We use these for deduplication, but you might find some use for them in doing Data Science.

The below functionality has a very low memory footprint (given the bigness of the data [768 x millions x float32]) by returning two separate numpy arrays - for Job Ad IDs and the vectors, respectively, such that the rows of IDs and vectors correspond one-to-one:

```python
from ojd_daps.dqa.data_getters import get_vectors
vectors, ids = get_vectors()

vectors
>>>
array([[ 0.30059,  0.64824,  0.20451, ..., -0.37065,  0.11025, -0.65322],
	   [ 0.30059,  0.64824,  0.20451, ..., -0.37065,  0.11025, -0.65322],
	   [ 0.1874 ,  0.11023,  0.22225, ...,  0.36673,  0.01616,  0.34427],
	   ...,
	   [ 0.00571,  0.59925,  0.22619, ...,  0.15325,  0.0813 ,  0.06835],
	   [ 0.20126,  0.91492,  0.69482, ...,  0.05599, -0.35545, -0.40659],
	   [ 0.02282,  0.35086,  0.52324, ...,  0.17844,  0.14136,  0.00848]],
	  dtype=float32)

ids
>>>
array(['41547517', '41547520', '41547521', ..., '41658879', '41658880',
	   '41658881'], dtype='<U40')

vectors.shape, ids.shape
>>>
((20000, 768), (20000,))
```

If you require fewer results, set `max_chunks` and `chunksize` for `max_chunks * chunksize` results.

## Contributing a Data Quality Analysis

### Steps:

1. Create an issue on GitHub, corresponding to the DQA which you about to perform. Create a branch and link a PR to your issue (as per the standard setup instructions on the main `README.md`).
2. Copy the `template_notebook.py` and give it a useful name (something like `{s3, db}_{<lower_camel_case_description>}.py`).
3. Do your analysis, trying to refactor utils up your notebook as you go
4. Make some plots
5. When you're finished, refactor your plots into `ojd_daps.dqa.plots.<your plot module>` (see the "rules" below)
6. Make sure any functions (other than `make_plot_data` and `make_plot`) which your plotting module rely on are unit-tested under the `tests` directory
7. Submit a PR

### Rules for the plotting module:

- The name should be `{s3, db}_{<lower_camel_case_description>}_{weekly, monthly, all}.py`
- Module contains function `make_plot_data` which outputs a iterable. There must be a `test` mode which runs in less than a few minutes.
- Module contains function `make_plot` which takes the output of `make_plot_data` and returns an `ax` object
- All plotting modules should be responsible for creating one plot.
