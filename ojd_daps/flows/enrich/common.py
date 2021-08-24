import itertools
from sqlalchemy.sql.expression import func


def flatten(iterable):
    return list(itertools.chain(*iterable))


def get_chunks(_list, chunksize):
    chunks = [_list[x : x + chunksize] for x in range(0, len(_list), chunksize)]
    return chunks


def generate_description_queries(flow, chunksize, min_text_len=5):
    """Generate raw SQL queries for job ad descriptions.
    The reason to do this is to avoid creating enormous artefacts
    in foreach steps, and instead reading one chunk of data per foreach step.

    Args:
        flow (FlowSpec): i.e. "self" in the flow
        chunksize (int): CHUNKSIZE in your flow module
    Returns:
        list of str: List of SQL queries, each query representing a chunk of data.
    """
    # Avoid errors in batch by importing here
    from ojd_daps.orms.raw_jobs import RawJobAd

    limit = 2 * chunksize if flow.test else None  # At least 2 chunks in test mode
    non_empty_text = func.length(RawJobAd.description) > min_text_len

    queries = []  # This is the output
    with flow.db_session(database="production") as session:
        # Retrieve all IDs in production DB
        query = session.query(RawJobAd.id)
        query = query.filter(RawJobAd.id is not None)
        query = query.filter(non_empty_text)
        ids = list(_id for _id, in query.limit(limit))
        # Generate one raw SQL query per chunk of IDs
        for chunk in get_chunks(ids, chunksize):
            query = session.query(
                RawJobAd.id, RawJobAd.data_source, RawJobAd.description
            )
            query = query.filter(RawJobAd.id.in_(chunk))
            raw_query = str(
                query.statement.compile(compile_kwargs={"literal_binds": True})
            )
            queries.append(raw_query)
    return queries


def retrieve_job_ads(flow):
    """Retrieve data from a raw SQL query. It is assumed that this
    call is made in a foreach step, and hence the query is stored in flow.input

    Args:
        flow (FlowSpec): i.e. "self" in the flow
    Returns:
        list of dict: The job ad data, including the job description
    """
    with flow.db_session(database="production") as session:
        results = session.execute(flow.input).fetchall()
    job_ads = [
        {"id": _id, "data_source": _src, "description": _desc}
        for _id, _src, _desc in results
    ]
    return job_ads
