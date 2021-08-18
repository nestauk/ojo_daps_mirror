#!/usr/bin/env bash
python surface_forms_flow.py run --production True --model_name v01_1 &&\
python batch_detection_flow.py run --production True --model_name v01_1 &&\
python remove_frequent_forms_flow.py run --detected_skills_path data/processed/detected_skills/sample_reed_extract_v01_1_skills.json --new_model_name v01_1 &&\
python notebooks/PIPELINE_assess_quality.py &&\
python notebooks/PIPELINE_general_skills.py &&\
python notebooks/PIPELINE_surface_form_clustering.py &&\
python notebooks/PIPELINE_cluster_assignments.py
