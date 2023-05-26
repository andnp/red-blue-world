#!/bin/bash
set -e

export PYTHONPATH=red_blue_world
python3 -m unittest discover -p "*test_*.py"
