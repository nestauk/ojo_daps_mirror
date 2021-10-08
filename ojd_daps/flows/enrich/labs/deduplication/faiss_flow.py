"""
deduplication.faiss_flow
-------------------

A flow for generating the faiss model.

Run this flow with

    python faiss_flow.py run
"""
from metaflow import FlowSpec, Parameter, step
from daps_utils import DapsFlowMixin

class FaissFlow(FlowSpec, DapsFlowMixin):
    k = Parameter(
        "k",
        help="Maximum number of duplicates that can be found",
        default='2500'
    )
    k_large = Parameter(
        "k_large",
        help="Sample size of neighbour documents",
        default='10000'
    )
    n_clusters = Parameter(
        "n_clusters",
        help="Number of clusters",
        default='250'
    )
    metric = Parameter(
        "metric",
        help="Metric used to define similarity",
        default='METRIC_L1'
    )
    score_threshold = Parameter(
        "score_threshold",
        help="Threshold at which similarity is deemed to be a duplicate",
        default='0.8'
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
        This isn't needed for this model, because it's parameter based
        """
        self.next(self.save_model)

    @step
    def save_model(self):
        """
        Pickle the model, or save model config
        """
        from faiss_utils import save_model
        save_model(
            k=self.k,
            k_large=self.k_large,
            n_clusters=self.n_clusters,
            metric=self.metric,
            score_threshold=self.score_threshold
            )
        self.next(self.end)

    @step
    def end(self):
        """
        Do any cleanup that's required
        """
        pass


if __name__ == "__main__":
    FaissFlow()
