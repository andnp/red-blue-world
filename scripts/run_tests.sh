#!/bin/bash
set -e

MYPYPATH=./typings mypy --ignore-missing-imports -p red_blue_world

export PYTHONPATH=red_blue_world
python3 -m unittest discover -p "*test_*.py"
