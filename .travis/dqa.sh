#!/usr/bin/env bash
set -e
TOPDIR=$PWD
pip install -r ojd_daps/dqa/requirements.txt
pip install awscli
pytest -x ojd_daps/dqa/tests/test_data_getters.py
pytest -x ojd_daps/dqa/tests/test_shared_cache*.py
