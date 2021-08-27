#!/usr/bin/env bash
set -e
TOPDIR=$PWD
pip install -r ojd_daps/dqa/requirements.txt
pytest -x ojd_daps/dqa/tests/test_data_getters.py
