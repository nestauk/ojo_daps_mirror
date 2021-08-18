"""
skills.batch_detection_flow
-------------------

A flow for detecting skills in a batch of texts (e.g. job adverts)

Run this flow with

    python batch_detection_flow.py run
"""
from metaflow import FlowSpec, Parameter, step
from daps_utils import DapsFlowMixin  # <-- Adds in a 'test' Parameter to your flow
from pathlib import Path
from ojd_daps.flows.enrich.labs.skills.skills_detection_utils import (
    load_model,
    detect_skills,
    clean_text,
    setup_spacy_model,
)
from helper_utils import save_json_to_s3, save_json


class BatchDetectionFlow(FlowSpec, DapsFlowMixin):
    """
    A flow to detect skills in a batch of texts
    """

    # Paths for inputs/outputs
    default_data_path = Path(__file__).parent.joinpath("data")
    text_data = Parameter( # Using Parameter here, to be able to access the original file name
        "text_data",
        help="Table with text data; requires columns 'id' and 'descriptions'",
        default=f"{default_data_path}/raw/job_ads/sample_reed_extract.csv",
    )
    text_column = Parameter("text_column", help="Column containg job descriptions", default="description")
    id_column = Parameter("id_column", help="Column containg job identifier", default="id")
    model_name = Parameter("model_name", help="Model name", default="")
    preprocessing = Parameter("preprocessing", help="If True, the flow will preprocess the text before extracting skills", default=True)

    @step
    def start(self):
        """Load the skills detection model, and text data"""
        import pandas as pd
        from io import StringIO

        # Subset the data for "test" mode
        sample_limit = 50 if self.test else None
        # Load skills detection model
        self.model = load_model(self.model_name)
        # Load text data
        data = pd.read_csv(self.text_data).iloc[:sample_limit]
        # Pull out the column with the unique identifier for each text
        self.text_id = data[self.id_column].to_list()
        # Pull out the column with the texts
        self.texts = data[self.text_column].to_list()
        self.next(self.detect_skills)

    @step
    def detect_skills(self):
        """Process skills"""
        # Preprocess the text data (if required)
        if self.preprocessing:
            texts = zip(self.text_id, (clean_text(s) for s in self.texts))
        else:
            texts = zip(self.text_id, self.texts)
        # Detect skills
        nlp=setup_spacy_model(self.model['nlp'])
        self.skills = {
            text_id: detect_skills(text, self.model, nlp=nlp, return_dict=True, debug=True)
            for text_id, text in texts
        }
        self.next(self.save_outputs)

    @step
    def save_outputs(self):
        """Save outputs to s3 and locally"""
        import json
        from pathlib import Path
        path_to_file = Path(self.text_data)
        filename = f"{str(path_to_file.stem)}_{self.model_name}_skills.json"
        outputs = {
            "file_name": path_to_file.name,
            "model_name": self.model_name,
            "detected_skills": self.skills,
        }
        save_json(outputs, f"data/processed/detected_skills/{filename}")
        save_json_to_s3(outputs, f"data/processed/detected_skills/{filename}")
        self.next(self.end)

    @step
    def end(self):
        """ """
        pass


if __name__ == "__main__":
    BatchDetectionFlow()
