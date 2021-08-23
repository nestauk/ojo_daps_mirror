#!/usr/bin/env bash
set -e
TOPDIR=$PWD
pytest -x ojd_daps/dqa/tests/test_data_getters.py
