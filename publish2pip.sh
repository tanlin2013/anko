#!/usr/bin/env bash

echo 'Removing old distribution'
rm -rf ./dist
python setup.py sdist
echo 'Connecting to pip'
twine upload dist/*