"""
salaries.regex.multi_regex_flow
-------------------

A flow for generating the regex model.
Note that the model is so simple that this flow is basically pointless.
Still, it's good to have it here for reproducibility.

Run this flow with

    python multi_regex_flow.py run
"""
from ojd_daps.flows.enrich.labs.salaries.regex.multi_regex_utils import save_model
from metaflow import FlowSpec, Parameter, step
from daps_utils import DapsFlowMixin  # <-- Adds in a 'test' Parameter to your flow


class RegexFlow(FlowSpec, DapsFlowMixin):
    regex = Parameter(
        "regex", help="The regex parameter for this model", default="(\d*[.]?\d*)"
    )

    @step
    def start(self):
        """
        Starts the flow, you could do some prep steps here
        """
        self.next(self.train_model)

    @step
    def train_model(self):
        """
        This isn't needed for this model, because it's so easy
        """
        # This would be a good place to subset the data for "test" mode
        # e.g.
        #
        # sample_limit = 100 if test else None
        # data = data[:sample_limit]
        #
        # but no data is needed for this model, so we can skip this requirement
        self.next(self.save_model)

    @step
    def save_model(self):
        """
        Pickle the model, or save model config
        """
        save_model(regex=self.regex)
        self.next(self.end)

    @step
    def end(self):
        """
        Do any cleanup that's required
        """
        pass


if __name__ == "__main__":
    RegexFlow()
