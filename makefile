.PHONY: clean-pyc clean-build all test clean

all: doc install publish

install: clean-build
	python setup.py install

test:
	python -m unittest discover -s test -p '*_test.py'

doxy_doc:
	doxygen Doxyfile

doc_rst:
	sphinx-apidoc -o docs/source/ anko/

doc:
	make -C docs html

publish: clean-dist
	python setup.py sdist
	echo 'Connecting to pip'
	twine upload dist/*

clean-dist:
	echo 'Removing old distribution'
	rm -rf ./dist
	
clean-build:
	echo 'Removing old build'
	rm -rf ./build
	
clean: clean-dist clean-build