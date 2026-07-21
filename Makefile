.PHONY: test-python test-web check

test-python:
	PYTHONPATH="$$PWD/src:$$PWD/deploy/src:$$PWD" pytest -q tests deploy/tests

test-web:
	npm --prefix apps/web run check

check: test-python test-web

