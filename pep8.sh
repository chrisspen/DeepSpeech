#!/bin/bash
VENV=${VENV:-.env}
$VENV/bin/pylint --rcfile=pylint.rc infer_server.py
