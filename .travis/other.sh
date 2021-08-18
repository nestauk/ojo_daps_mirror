#!/usr/bin/env bash
set -e

pytest -x ojd_daps/orms
pytest -x ojd_daps/tasks
