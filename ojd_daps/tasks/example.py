import luigi
import re
from daps_utils import CurateTask

from ojd_daps.orms.example_orm import People

"""Pre-compile the people ID regex for better performance"""
PEOPLE_ID_REGEX = re.compile('http://swapi.dev/api/people/(.*)/')


def get_id_from_url(url):
    id_str, = PEOPLE_ID_REGEX.findall(url)  # Only expect one
    return int(id_str)


class ExampleCurateTask(CurateTask):
    def curate_data(self, s3path):
        data = self.retrieve_data(s3path, 'swapi-people')
        # Filter out columns we don't want
        bad_fields = ['films', 'species', 'vehicles',
                      'starships', 'edited', 'skin_color'] 
        data = [{k: v for k, v in row.items() if k not in bad_fields} for row in data]
        # Convert birth_year to a number, and generate id
        for row in data:
            row['id'] = get_id_from_url(row['url'])
            row['birth_year'] = int(''.join(i for i in row['birth_year'] if i.isnumeric()))
        return data


class RootTask(luigi.WrapperTask):
    def requires(self):
        return ExampleCurateTask(orm=People,
                                 flow_path='examples/example.py',
                                 preflow_kwargs={'datastore': 's3',
                                                 'package-suffixes': '.txt'})

