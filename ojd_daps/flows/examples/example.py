"""
example_flow
------------

An example Flow for writing to S3, interacting with batch,
and is able to be wrapped up in a "luigi" MetaflowTask. The task itself
will collect data from the Star Wars API (swapi), and the task
can collect different types of entity, depending on what you specify.

To run this:

    python example.py --package-suffixes=.txt --datastore=s3 run

Explanation:

    * package-suffixes=.txt will include the local requirements.txt in the AWS batch bundle
    * datastore=s3 is stipulated by metaflow when using the @batch decorator, so it can write to somewhere!
"""
from daps_utils import talk_to_luigi
import requests_cache
import numpy as np

from metaflow import FlowSpec, step, S3
from metaflow import Parameter, batch, pip
import json
import requests

ALLOWABLE_CODES = (404, 200)
requests_cache.install_cache(
    "swapi_cache",
    backend="sqlite",
    allowable_codes=ALLOWABLE_CODES,
    allowable_methods=("GET", "HEAD"),
)


def halfway(a, b):
    """Return an integer halfway between `a` and `b`. In the case of a "tie",
    `b` is returned, via `ceil`"""
    return int(np.ceil((a + b) / 2))


def make_swapi_url(swapi_type, number):
    """Construct the SWAPI query URL"""
    return f"https://swapi.dev/api/{swapi_type}/{number}/"


def request_row_as_json(swapi_type, number):
    """Make the SWAPI query"""
    url = make_swapi_url(swapi_type, number)
    r = requests.get(url)
    return r.json()


def page_exists(swapi_type, number):
    """Perform a HEAD request to see if the given item exists."""
    url = make_swapi_url(swapi_type, number)
    r = requests.head(url)
    if r.status_code not in ALLOWABLE_CODES:
        r.raise_for_status()
    return r.status_code == 200  # status_code will be 200 or 404


def find_last_page(swapi_type, lower_lim=0, upper_lim=100):
    """Uses a binary search to find the maximum count of `swapi_type` object"""
    # Update the limits based on whether the page is found or not
    attempt = halfway(lower_lim, upper_lim)
    if page_exists(swapi_type, attempt):
        lower_lim = attempt
    else:
        upper_lim = attempt
    # The following means that we've found the true upper limit
    # (noting that `halfway(...)` returns *ceil*)
    if lower_lim == upper_lim - 1:
        return lower_lim
    # If not found, try agains
    return find_last_page(swapi_type, lower_lim, upper_lim)


def generate_page_numbers(first_page, nominal_last_page, actual_last_page):
    """Resolve what the first and last page should be, making sure that there
    aren't any logical conflicts."""
    last_page = nominal_last_page  # Nominally, take the nominal_last_page
    # Default to actual_last_page if a nominal page wasn't provided OR
    # nominal_last_page is outside of the page limits
    if nominal_last_page is None or nominal_last_page > actual_last_page:
        last_page = actual_last_page
    # If this isn't true, we've got problems
    if first_page > last_page:
        raise ValueError(
            f"First page ({first_page}) is greater " f"than last page ({last_page})"
        )
    # Generate page numbers for batch iteration
    return range(first_page, last_page)


@talk_to_luigi
class BatchDemoFlow(FlowSpec):
    production = Parameter("production", help="Run in production mode?", default=False)
    swapi_type = Parameter(
        "swapi_type",
        help="One of 'people', 'planets' and 'starships'",
        default="people",
    )
    first_page = Parameter("first_page", help="First API page to hit", default=1)
    last_page = Parameter("last_page", help="Last API page to hit", default=None)

    @property
    def test(self):
        return not self.production

    @step
    def start(self):
        """We don't know a-priori what the last page is in the API,
        so rather than use brute-force, we use our brains. Using a binary
        search tree, we are able to determine the last page of the API
        in log(N) time. We then run 'foreach' via batch over the page range."""
        actual_last_page = find_last_page(
            self.swapi_type
        )  # Find the last page number of the API
        nominal_last_page = (
            self.first_page + 2
            if self.test  # Restrict pages in test mode
            else self.last_page
        )  # and otherwise take the provided last page number
        self.pages = generate_page_numbers(
            first_page=self.first_page,
            nominal_last_page=nominal_last_page,
            actual_last_page=actual_last_page,
        )
        self.next(self.get_data, foreach="pages")

    @batch
    @pip(path="requirements.txt")
    @step
    def get_data(self):
        """Get the row of data for this page number, noting that
        'self.input' is the page number."""
        self.row = request_row_as_json(self.swapi_type, self.input)
        self.next(self.join_data)

    @step
    def join_data(self, inputs):
        """Method required by @batch steps for joining outputs back together again"""
        self.data = [input.row for input in inputs]
        self.next(self.end)

    @step
    def end(self):
        """Save the data to the data lake"""
        filename = f"swapi-{self.swapi_type}_test-{self.test}.json"  # NB: output path is defined from Flow parameters
        with S3(run=self) as s3:
            data = json.dumps(self.data)
            url = s3.put(filename, data)


if __name__ == "__main__":
    BatchDemoFlow()
