#!/usr/bin/env bash

echo 'New version number: '
read ver
echo 'Removing old distribution'
rm -rf ./dist
python setup.py sdist $ver
echo 'Connecting to pip'
twine upload dist/*