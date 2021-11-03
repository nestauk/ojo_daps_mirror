#!/bin/bash
################################################################
### Text automatically added by daps-utils calver-init ###
set -e
chmod +x .githooks/pre-commit
bash .githooks/pre-commit
pip install -e . -r requirements-dev.txt
pre-commit install --install-hooks
echo "os.execvp(\"bash\", [\"bash\", \"${PWD}/.githooks/pre-commit\"])" >> .git/hooks/pre-commit
git config --local include.path ../.gitconfig
pre-commit run --all-files
echo install.sh completed successfully
################################################################
