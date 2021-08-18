#!/bin/bash
################################################################
### Text automatically added by daps-utils calver-init ###
set -e
chmod +x .githooks/pre-commit
bash .githooks/pre-commit
ln -sf $PWD/.githooks/pre-commit .git/hooks/pre-commit
git config --local include.path ../.gitconfig
echo install.sh completed successfully
################################################################
