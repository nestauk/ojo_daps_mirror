## Getting started

Make sure that you have followed all of the guidelines in the [core README](https://github.com/nestauk/ojd_daps/blob/dev/README.md#for-contributors).

## Notebooks

Notebooks are not versioned in this repo (they are .gitignored), but that doesn't mean you shouldn't use them. You should use [jupytext](https://github.com/mwouts/jupytext) for converting your notebooks into standard python files.

_Please do not force your notebook to be committed_, please stick to using jupytext. Our reason for this "rule" is that jupytext "notebooks" are substantially smaller than standard `jupyter` notebooks and they can be reliably git diff'd (and therefore reviewed).

However _please do commit your jupytext "notebook"_: the history of these is a valuable contribution to this repo!

## Coding style and Git conventions

You must follow our git conventions - this is non-negotiable.

Please follow the data science coding style guide, which is generally a simplication of the long-standing industry standard ["pep8"](https://www.python.org/dev/peps/pep-0008/) plus uncontroversial patterns that most people converge on with experience.

Remember - we're striving to become better coders: make sure to remove repetitions of code, add clear comments and docstrings to make it easier to follow the logic, and write unit tests for your functions. Generally, keep your code efficient and avoid inelegant code by following the style guide. Your PR buddy will also help you with this, see below.

- [Git conventions](https://github.com/nestauk/ojd_daps/blob/dev/README.md#contribution-etiquette)
- [Coding style](https://github.com/nestauk/cookiecutter-data-science-nesta/blob/master/STYLE_GUIDE.md)

## PRs and PR buddies

With every PR you will assign a reviewer who already understands the codebase and the required standards. In general this should be someone who has merged at least a couple of PRs in this repo. For your first couple of PRs in this repo your PR reviewer will be your PR buddy: they are expecting that you might need some assistance along the way with any questions or stumbling blocks. If you don't have a PR buddy then get one! If you need help finding a buddy, then ask in the `dev` channel on Slack- we will be happy to assign you a buddy.

PR buddies (and more broadly to all reviewers): you are the health inspector. You _must_:

- Ensure the git conventions and coding style guide are adhered to.
- Be an annoying nitpicking know-it-all - that is your core job. If there is a difference in opinion then raise it with the Data Engineering team, we're happy to settle debates.
- Run the code in **at least** testing-mode. You should be able to reproduce results as presented in the "notebook".

You _musn't_ accept a PR:

- Without (re)generating a model.
- Without verifying that you can load and apply the model, in one step (see `salaries.regex`) in the form `def f(dict): return prediction`
- If there is any repetition in the code, or there are glaring inefficiencies, or unexplained sections.
- If functions are too large (see [Single Responsibility Principle](https://en.wikipedia.org/wiki/Single-responsibility_principle))"
- If functions have not been unit-tested
- If there is no _testing_ mode for running the code
  - i.e. a mode flag for running your code from start-to-finish ("end-to-end") in a faster manner (e.g. by using a smaller subset of data).

If you are new to PR buddying then ask the Data Engineering team! We will be happy to assign you someone to hold your virtual hand for your first PR(s), noting that _you_ will be leading the PR.

## So... what do I need to do?

tldr; create a model (e.g. to predict salaries), write a function for loading and applying the model to a row of data (`dict`) to return a feature (e.g. the salary). Put the code for generating the model in a flow.

From this directory you will see one folder per feature, at time of writing these were:

- `job_titles`
- `salaries`
- `sic`
- `skills`
- `soc`

Over time, we will have multiple methods for extracting e.g. salaries from the raw job adverts. You should give your method a **short, simple and descriptive name**, such as `random_forests` or `regex`. Under `salaries` you will see:

```
.
├── tests/          <-- at least one test per function in `regex_utils`
├── regex.py        <-- your jupytext notebook, for prototyping
├── regex_flow.py   <-- generates a model using `regex_utils` ONLY
└── regex_utils.py
```

where `regex_utils.py` contains all the pieces prototyped in `regex.py` for:

- Generating a model, or model parameters.
- Loading and applying the model in a single step, in the form `def f(dict): return predicted_feature`

Note from `regex_utils.py` that you should load models using the `lru_cache` decorator. If it isn't clear to you why this is then ask!

Models, model parameters or any other metadata should be stored in the `open-job-lake` in the form `s3://open-jobs-lake/labs/<feature name>/<method name> `, for example `s3://open-jobs-lake/labs/salaries/regex`.

## Running tests:

Make sure your `PYTHONPATH` is set to `/path/to/ojd_daps` (note, not `/path/to/ojd_daps/ojd_daps`) and run `pytest path/to/tests`. For example:

```python
cd path/to/ojd_daps
export PYTHONPATH=$PWD
cd ojd_daps/flows/enrich/labs
pytest salaries/tests
```

of course, assuming you have already set up your `conda` environment.

## Additional requirements

If your code needs additional requirements, please include these in your labs directory e.g. `ojd_daps/flows/enrich/labs/salaries/requirements.txt`

## Example: `salaries.regex`

Save the model like:

```python
save_model('(\d*[.]?\d*)', 'max')  # <--- After all of my hard work, I'll save my model config
```

Load the model like:

```python
@lru_cache()  # <--- Important
def load_model():
    """Loads the model"""
    regex = load_from_s3('regex.txt')
    picker_name = load_from_s3('picker.txt')
    return regex_model(regex, picker_name)
```

Apply the model like

```python
def apply_model(row):
    """Loads and applies the model to the given row"""
    model = load_model()  # NB: lru_cached
    salary = model(row['job_salary_raw'])
    rate = guess_rate(salary)
    return salary, rate
```

and if you want to see the results on real data:

```python
def load_jobs(limit=10):
    with db_session('production') as session:
        for ad in session.query(RawJobAd).limit(limit):
            yield object_as_dict(ad)

# Example of applying my model
fields_to_print = ('job_title_raw', 'contract_type_raw', 'job_salary_raw')
for job_ad in load_jobs():
    prediction = apply_model(job_ad)
    print(*((x, job_ad[x]) for x in fields_to_print), ('prediction -->', prediction), sep="\n")
    print()
```

output:

```
('job_title_raw', 'Logistics Operative')
('contract_type_raw', 'Temporary')
('job_salary_raw', '9.0000-9.5000')
('prediction -->', (9.5, 'per hour'))

('job_title_raw', 'Customer Assistant Part Time')
('contract_type_raw', 'Permanent')
('job_salary_raw', '9.3000-10.5000')
('prediction -->', (10.5, 'per hour'))

('job_title_raw', 'Senior Java Developer - SC Cleared')
('contract_type_raw', 'Permanent')
('job_salary_raw', '50000.0000-75000.0000')
('prediction -->', (75000.0, 'per annum'))

('job_title_raw', 'Account Manager - Oxford')
('contract_type_raw', 'Permanent')
('job_salary_raw', '65000.0000-65000.0000')
('prediction -->', (65000.0, 'per annum'))

('job_title_raw', 'Teacher of Mathematics')
('contract_type_raw', 'Contract')
('job_salary_raw', '138.5600-212.5200')
('prediction -->', (212.52, 'per day'))

('job_title_raw', 'Mathematics Teacher')
('contract_type_raw', 'Contract')
('job_salary_raw', '138.5600-212.5200')
('prediction -->', (212.52, 'per day'))

('job_title_raw', 'Nursing Home Manager')
('contract_type_raw', 'Permanent')
('job_salary_raw', '53655.0000-53655.0000')
('prediction -->', (53655.0, 'per annum'))

('job_title_raw', 'Commercial Finance Business Partner - Relocation Required')
('contract_type_raw', 'Permanent')
('job_salary_raw', '50000.0000-60000.0000')
('prediction -->', (60000.0, 'per annum'))

('job_title_raw', 'SEN Teacher')
('contract_type_raw', 'Permanent')
('job_salary_raw', '130.0000-200.0000')
('prediction -->', (200.0, 'per day'))

('job_title_raw', 'Social Worker - Assessment')
('contract_type_raw', 'Temporary')
('job_salary_raw', '32.0000-32.0000')
('prediction -->', (32.0, 'per hour'))

```
