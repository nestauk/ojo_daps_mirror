"""
Common functions
"""

import itertools


def flatten(iterable):
    return list(itertools.chain(*iterable))


def get_chunks(_list, chunksize):
    chunks = [_list[x : x + chunksize] for x in range(0, len(_list), chunksize)]
    return chunks
