#!/usr/bin/env bash
python notebooks/PIPELINE_assess_quality.py &&\
python notebooks/PIPELINE_general_skills.py &&\
python notebooks/PIPELINE_surface_form_clustering.py &&\
python notebooks/PIPELINE_cluster_assignments.py
