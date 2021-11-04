# Example

This is an example flow, which can also be run from the luigi `tasks/example.py` routine, which curates the data
and adds it to S3.

The flow itself will collect data from the Star Wars API (swapi). The flow
can collect different types of entity, depending on what you specify (show the flow help for more info!).

To run this:

    python example.py --package-suffixes=.txt --datastore=s3 run

Explanation:

    * package-suffixes=.txt will include the local requirements.txt in the AWS batch bundle
    * datastore=s3 is stipulated by metaflow when using the @batch decorator, so it can write to somewhere!
