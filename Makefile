.PHONY: all test smoke reproduce doctor clean install

all: test smoke

test:
	python3 -m unittest discover tests

smoke:
	python3 -m resonance.smoke

doctor:
	python3 -m resonance.cli doctor

reproduce:
	python3 scripts/render_readme_tables.py --check

install:
	pip install -e .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
