"""
SOC lookup
----------

A Flow for generating a lookup table of SOC job titles to SOC codes
"""
import json
from hashlib import md5

from daps_utils import DapsFlowMixin

from metaflow import FlowSpec, S3, step

from ojd_daps.flows.enrich.labs.soc.metadata_utils import (
    load_json_from_s3,
    save_metadata_to_s3,
)


def short_hash(text):
    """Generate a unique short hash for this string"""
    hx_code = md5(text.encode()).hexdigest()
    int_code = int(hx_code, 16)
    short_code = str(int_code)[:16]
    return int(short_code)


class SocMetadataFlow(FlowSpec, DapsFlowMixin):
    @step
    def start(self):
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
