"""
locations.regex_flow
-------------------
A flow for generating the regex model.
Note that the model is so simple that this flow is basically pointless.
    python regex_flow.py run
"""
from regex_utils import save_model
from metaflow import FlowSpec, Parameter, step


class RegexFlow(FlowSpec):
    outcode_regex = Parameter(
        "ouctode_regex",
        help="The postcode area regex parameter for this model",
        default="[A-Z]{1,2}[0-9][0-9A-Z]?\s?",
    )
    boilerplate_text = Parameter(
        "boilerplate_text",
        help="The cleaning regex parameter for this model",
        default="[^\w\s]",
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
        self.next(self.save_model)

    @step
    def save_model(self):
        """
        Pickle the model, or save model config
        """
        save_model(
            outcode_regex=self.outcode_regex, boilerplate_text=self.boilerplate_text
        )
        self.next(self.end)

    @step
    def end(self):
        """
        Do any cleanup that's required
        """
        pass


if __name__ == "__main__":
    RegexFlow()
