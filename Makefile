.PHONY: man test up check dist

test:
	pytest

check:
	restview --long-description --strict

dist:
	sudo python setup.py bdist_wheel

up:
	twine upload dist/`ls dist -rt | tail -1`



