"""
skills.surface_forms_flow
-------------------

A flow for generate phrases that serve as 'surface forms'
and have one-to-one relation to specific ESCO skills 'entities'.

Run this flow with

    python surface_forms_flow.py run --production False --model_name <model_name>
"""
from metaflow import FlowSpec, Parameter, step, IncludeFile
from daps_utils import DapsFlowMixin  # <-- Adds in a 'test' Parameter to your flow
from pathlib import Path

from ojd_daps.flows.enrich.labs.skills.skills_detection_utils import (
    setup_spacy_model,
    clean_text,
    split_string,
    flatten_skills_labels,
    create_documents,
    create_surface_forms,
    filter_rows,
    surface_form_dataframe,
    deduplicate_surface_forms,
    remove_and_reassign_forms,
    tfidf_vectorise,
    tfidf_phrase_sums,
    tfidf_representativity_mask,
    create_phrase_matcher,
    save_model_in_s3,
    save_model_locally,
    save_removed_forms,
    DEF_LANGUAGE_MODEL,
)


class SurfaceFormsFlow(FlowSpec, DapsFlowMixin):
    """
    A flow to generate surface forms from ESCO skills data
    """

    # Paths for inputs/outputs
    default_data_path = Path(__file__).parent.joinpath("data")
    data_path = Parameter(
        "data_path", help="Path to the data folder", default=default_data_path
    )
    model_name = Parameter(
        "model_name", help="Suffix to use when saving the model", default=""
    )
    # Input data
    esco_skills = IncludeFile(
        "esco_skills",
        help="Table with ESCO skills, their labels and descriptions",
        default=f"{default_data_path}/raw/esco/ESCO_skills_hierarchy.csv",
    )
    # Parameters & inputs for manual tweaks
    surface_form_types = Parameter(
        "surface_form_types",
        help="Different types of surface forms to generate (in the order of importance/reliability)",
        default=["label_pref", "label_alt", "chunk_pref", "chunk_alt", "chunk_descr"],
    )
    min_length = Parameter(
        "min_length", help="Minimal length for surface forms", default=3
    )
    manual_adjustments_file = IncludeFile(
        "manual_adjustments_file",
        help="Dictionary with surface forms that have been manually selected for keeping or removing",
        default=f"{default_data_path}/aux/manual_adjustments.json",
    )

    @step
    def start(self):
        """
        Ingest a CSV with skills from the European Skills, Competences,
        Qualifications and Occupations (ESCO), prepare the input skills data,
        and launch parallel surface form generators for each surface form type
        """
        import pandas as pd
        from io import StringIO
        import json

        # Subset the data for "test" mode
        sample_limit = 50 if self.test else None

        # Load the ESCO skills table into a pandas DataFrame;
        # each row in this table corresponds to one ESCO skill entity
        self.skills = pd.read_csv(StringIO(self.esco_skills))
        self.skills = self.skills.iloc[:sample_limit]

        # Each skill entity has its main so-called 'preferred label' and a description that
        # is one or more sentences long. Pull out the columns corresponding to both
        # of these features for subsequent processing, and keep track of the entity integer identifier
        self.entities_id = self.skills.id.to_list()
        self.labels_preferred = self.skills.preferred_label.to_list()
        self.descriptions = self.skills.description.to_list()

        # Each skills entity has an additional set of alternate labels, which are
        # all stored in one string, and separated by a newline symbol.
        alternate_labels = self.skills.alt_labels.to_list()
        # "Flatten" this list of alternate labels and, for each label, keep track
        # of this corresponding entity identifier
        self.entities_id_alt, self.labels_alternate = flatten_skills_labels(
            self.entities_id, alternate_labels
        )

        # Load instructions for manual adjustments, describing which surface forms to
        # manually remove, keep or reassign to different entities. These adjustments
        # have been determined by manually reviewing the results in the prototyping stage.
        self.manual_adjustments = json.loads(self.manual_adjustments_file)

        # Store surface forms that will be removed during refinement steps,
        # so that it is possible to review them later
        self.removed_forms_dict = {}

        self.form_types = self.surface_form_types
        self.next(self.generate_surface_forms, foreach="form_types")

    @step
    def generate_surface_forms(self):
        """
        Generate surface forms for each skills entity (fan out)
        """
        self.form_type = self.input
        self.forms = None

        # Generate surface forms from the specified input data
        if "pref" in self.form_type:
            # Generate surface forms from skill preferred labels
            self.forms = create_surface_forms(
                self.entities_id,
                self.labels_preferred,
                chunk=("chunk" in self.form_type),
            )
        elif "alt" in self.form_type:
            # Generate surface forms from skill alternate labels
            self.forms = create_surface_forms(
                self.entities_id_alt,
                self.labels_alternate,
                chunk=("chunk" in self.form_type),
            )
        elif "descr" in self.form_type:
            # Generate surface forms from skill descriptions
            self.forms = create_surface_forms(
                self.entities_id, self.descriptions, chunk=("chunk" in self.form_type)
            )

        self.next(self.join)

    @step
    def join(self, inputs):
        """
        Join the parallel branches, merge results into a dataframe,
        and remove duplicate surface forms.
        """
        # Collect inputs
        surface_forms = {inp.form_type: inp.forms for inp in inputs}
        self.merge_artifacts(inputs, exclude=["form_type", "forms"])
        # Create a dataframe with deduplicated surface forms
        self.surface_forms = deduplicate_surface_forms(
            surface_form_dataframe(surface_forms), sort_order=self.form_types
        )
        self.next(self.refine_by_length)

    @step
    def refine_by_length(self):
        """Remove very short surface forms"""
        # Check witch surface forms have sufficiently many characters
        sufficiently_long = list(
            self.surface_forms.surface_form.str.len() >= self.min_length
        )
        # Remove surface forms that are too short
        self.surface_forms, removed_forms = filter_rows(
            self.surface_forms,
            rows_to_keep=sufficiently_long,
            forms_to_keep=self.manual_adjustments["keep"],
        )
        # Keep track of the removed surface forms
        self.removed_forms_dict["refine_by_length"] = removed_forms
        self.next(self.refine_by_relevance)

    @step
    def refine_by_relevance(self):
        """
        Use tf-idf vectors to ascertain surface form representativeness.
        """
        # Create and preprocess 'skills documents' for tf-idf analysis
        document_generator = create_documents(
            (
                self.labels_preferred,
                [" ".join(split_string(s)) for s in self.skills.alt_labels.to_list()],
                self.descriptions,
            )
        )
        skills_docs = [clean_text(s) for s in document_generator]
        # Vectorise the skills documents
        tfidf_matrix, vocabulary = tfidf_vectorise(skills_docs)
        # Estimate surface form representativity
        sufficiently_representative = tfidf_representativity_mask(
            self.surface_forms, tfidf_matrix, vocabulary
        )
        # Surface forms that are not "most representative" of their entities are removed
        self.surface_forms, removed_forms = filter_rows(
            self.surface_forms,
            rows_to_keep=sufficiently_representative,
            forms_to_keep=self.manual_adjustments["keep"],
        )
        self.removed_forms_dict["refine_by_relevance"] = removed_forms
        self.next(self.refine_by_manual_adjustments)

    @step
    def refine_by_manual_adjustments(self):
        """Remove and add manually selected surface forms"""
        self.surface_forms = remove_and_reassign_forms(
            self.surface_forms, self.manual_adjustments
        )
        self.next(self.create_model)

    @step
    def create_model(self):
        """Create a spacy phrase matcher"""
        nlp = setup_spacy_model()
        self.matcher = create_phrase_matcher(
            self.surface_forms.surface_form.to_list(), nlp
        )
        self.next(self.save_model)

    @step
    def save_model(self):
        """Save the outputs: surface form table and spacy phrase matcher"""
        # Add surface form entity labels
        self.surface_forms = self.surface_forms.merge(
            self.skills[["id", "preferred_label"]],
            left_on="entity",
            right_on="id",
            how="left",
            validate="m:1",
        ).drop("id", axis=1)
        # Create the model dictionary
        self.model = {
            "name": self.model_name,
            "surface_forms": self.surface_forms,
            "matcher": self.matcher,
            "nlp": DEF_LANGUAGE_MODEL,
        }
        # Save the model
        save_model_in_s3(self.model)
        save_model_locally(self.model)
        self.next(self.end)

    @step
    def end(self):
        """Save any other data for a later review and end"""
        save_removed_forms(self.removed_forms_dict, self.model_name)


if __name__ == "__main__":
    SurfaceFormsFlow()
