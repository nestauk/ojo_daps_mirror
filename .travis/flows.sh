#!/usr/bin/env bash
set -e
TOPDIR=$PWD

BASE_PYTHONPATH=$PYTHONPATH
FLOWDIR=ojd_daps/flows
for DIRNAME in $(ls $FLOWDIR);
do
    # Save the venv base for later
    cd $TOPDIR
    cp -r venv venv-copy
    # Go to the flow dir and install any local requirements
    cd $TOPDIR/$FLOWDIR/$DIRNAME
    echo $PWD
    ls requirements.txt &> /dev/null && pip install -r requirements.txt || echo "no requirements.txt to install"
    ls requirements_test.txt &> /dev/null && pip install -r requirements_test.txt || echo "no requirements.txt to install"
    # Run the tests
    PYTHONPATH=$BASE_PYTHONPATH:$PWD pytest -x .
    # Revert back the base venv
    cd $TOPDIR
    rm -rf venv
    mv venv-copy venv
done
