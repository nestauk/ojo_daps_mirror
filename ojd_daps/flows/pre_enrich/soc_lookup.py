"""
SOC lookup
----------

A Flow for generating a lookup table of SOC job titles to SOC codes
"""
import os

for reqs in ["", "_soc"]:
    os.system(
        f"pip install -r {os.path.dirname(os.path.realpath(__file__))}/"
        f"requirements{reqs}.txt 1> /dev/null"
    )

import json
from hashlib import md5
from metaflow import FlowSpec, step, S3
from daps_utils import talk_to_luigi, DapsFlowMixin

from ojd_daps.flows.enrich.labs.soc.metadata_utils import (
    save_metadata_to_s3,
    load_json_from_s3,
)


def short_hash(text):
    """Generate a unique short hash for this string"""
    hx_code = md5(text.encode()).hexdigest()
    int_code = int(hx_code, 16)
    short_code = str(int_code)[:16]
    return int(short_code)


@talk_to_luigi
class SocMetadataFlow(FlowSpec, DapsFlowMixin):
    @step
    def start(self):
        """
        Starts the flow.
        """
        save_metadata_to_s3()
        self.next(self.end)

    @step
    def end(self):
        """Upload the SOC lookup"""
        soc_to_title = load_json_from_s3("soc_code_to_std_title")
        soc_lookups = [
            {
                "soc_id": short_hash(soc_code + standard_title),
                "soc_code": soc_code,
                "soc_title": standard_title,
            }
            for soc_code, standard_titles in soc_to_title.items()
            for standard_title in standard_titles
        ]
        # Validate that SOC IDs are unique
        soc_ids = [lookup["soc_id"] for lookup in soc_lookups]
        if len(soc_ids) > len(set(soc_ids)):
            raise ValueError("Non-unique hashes found")
        # All good, save to S3
        with S3(run=self) as s3:
            filename = f"soc_lookup-test-{self.test}.json"
            data = json.dumps(soc_lookups)
            s3.put(filename, data)


if __name__ == "__main__":
    SocMetadataFlow()
