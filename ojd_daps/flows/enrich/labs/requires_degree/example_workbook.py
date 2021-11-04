#!/usr/bin/env python
"""Example of applying requires_degree enrichment to job ads. Basic regex model that
returns True if any common degree-related word (e.g., Bachelor's) or acronym (BA) is in
the job description."""
from ojd_daps.orms.raw_jobs import RawJobAd
import model


# Save model regex to S3 (default regex pattern)
model.save_model()

if __name__ == "__main__":
    # number of jobs to apply model to, and number of positive cases to print
    # descriptions for
    N_TEST = 10
    N_PRINT = 3
    # Example of applying my model
    print("\nExample application of model:")
    fields_to_print = ("job_title_raw",)
    for job_ad in model.io.load_jobs(limit=N_TEST):
        prediction = model.apply_model(job_ad)
        print(*(job_ad[x] for x in fields_to_print), prediction, sep="\n")
        print()

    # few jobs require a degree, so let's print one that does
    print("\nExample jobs *with* degree requirement:")
    n_printed = 0
    for job_ad in model.io.load_jobs(
        limit=10000, columns=(RawJobAd.id, RawJobAd.description)
    ):
        if model.apply_model(job_ad):
            print()
            print(job_ad["id"])
            print(job_ad["description"])
            n_printed += 1
            if n_printed >= N_PRINT:
                break
