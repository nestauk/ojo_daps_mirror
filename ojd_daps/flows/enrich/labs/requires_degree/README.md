# Requires Degree
This enrichment detects whether a job requires a degree by detecting degree-related
terms in the description.

## Key files

`example_workbook.py`: Demonstration of feature in action

`model/__init__.py`: Main API for regex model (`load_model`, `save_model`, `apply_model`)

`model/io.py`: s3 and db interface

`model/nlp.py`: text processing
