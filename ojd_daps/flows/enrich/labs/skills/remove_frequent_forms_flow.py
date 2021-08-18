"""
skills.remove_frequent_forms_flow
-------------------

A flow for checking most frequent surface forms, and removing them from the model

Run this flow with

    python remove_frequent_forms_flow.py run
"""
from metaflow import FlowSpec, Parameter, step, IncludeFile
from daps_utils import DapsFlowMixin  # <-- Adds in a 'test' Parameter to your flow

from skills_detection_utils import (
    load_model,
    load_removed_forms,
    setup_spacy_model,
    filter_rows,
    count_surface_forms,
    frequency_refinement,
    create_phrase_matcher,
    save_model_in_s3,
    save_model_locally,
    save_removed_forms,
)
from pathlib import Path


class RemoveFrequentSurfaceFormsFlow(FlowSpec, DapsFlowMixin):
    """
    A flow to remove very frequent forms from the model
    """

    # Paths for input data
    default_data_path = Path(__file__).parent.joinpath("data")
    detected_skills_path = Parameter(  # Using Parameter here, to be able to access the original file name
        "detected_skills_path",
        help="Table with text data; requires columns 'id' and 'descriptions'",
        default=f"{default_data_path}/processed/detected_skills/sample_reed_extract__skills.json",
    )
    # Parameters & inputs for manual tweaks
    percentile_threshold = Parameter(
        "percentile_threshold",
        help="Percentile above which to remove the skills",
        default=95,
    )
    manual_adjustments_file = IncludeFile(
        "manual_adjustments_file",
        help="Dictionary with surface forms that have been manually selected for keeping or removing",
        default=f"{default_data_path}/aux/manual_adjustments.json",
    )
    new_model_name = Parameter("new_model_name", help="Model name", default="refined")

    @step
    def start(self):
        """Load the detected skills"""
        import json

        # Load the detected skills
        self.detected_skills = json.load(open(self.detected_skills_path, "r"))
        self.model_name = self.detected_skills["model_name"]
        # Load the model
        self.model = load_model(self.model_name)
        # Load manual adjustments
        self.manual_adjustments = json.loads(self.manual_adjustments_file)
        # Store removed surface forms to review later
        self.removed_forms_dict = load_removed_forms(self.model_name)
        self.next(self.remove_most_frequent_forms)

    @step
    def remove_most_frequent_forms(self):
        """Find and remove the most frequent forms"""
        # Count surface form occurrences
        counts = count_surface_forms(
            list(self.detected_skills["detected_skills"].values())
        )
        # Determine the surface from refinement based on their frequency
        rows_to_keep = frequency_refinement(
            self.model["surface_forms"], counts, self.percentile_threshold
        )
        # Filter the surface form table
        self.surface_forms, removed_forms = filter_rows(
            self.model["surface_forms"],
            rows_to_keep=rows_to_keep,
            forms_to_keep=self.manual_adjustments["keep"],
        )
        # Keep track of the removed surface forms
        self.removed_forms_dict["remove_most_frequent_forms"] = removed_forms
        self.next(self.create_model)

    @step
    def create_model(self):
        """Create a spacy phrase matcher"""
        nlp = setup_spacy_model(self.model["nlp"])
        self.matcher = create_phrase_matcher(
            self.surface_forms.surface_form.to_list(), nlp
        )
        self.next(self.save_model)

    @step
    def save_model(self):
        """Save the outputs: surface form table and spacy phrase matcher"""
        # Create the model dictionary
        self.new_model = {
            "name": self.new_model_name,
            "surface_forms": self.surface_forms,
            "matcher": self.matcher,
            "nlp": self.model["nlp"],
        }
        # Save the model
        save_model_in_s3(self.new_model)
        save_model_locally(self.new_model)
        self.next(self.end)

    @step
    def end(self):
        """Save the removed forms and end"""
        save_removed_forms(self.removed_forms_dict, self.new_model_name)


if __name__ == "__main__":
    RemoveFrequentSurfaceFormsFlow()
