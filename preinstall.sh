#!/bin/bash
# Pré-compiler pystan
pip install --no-cache-dir --only-binary :all: pystan==2.19.1.1

# Installer les autres dépendances
pip install --no-cache-dir -r requirements.txt
