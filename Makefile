.PHONY: test coverage coverage-html coverage-check lint install

PY := venv/bin/python
PIP := venv/bin/pip

install:
	$(PIP) install -r requirements.txt

test:
	$(PY) -m pytest -q

coverage:
	$(PY) -m pytest tests/format \
		--cov=src/catan_llm/format \
		--cov-report=term-missing \
		--cov-report=html \
		--cov-report=xml

coverage-html: coverage
	@echo "Open htmlcov/index.html"

# Enforce thresholds — mirrors CI (ci.yml)
coverage-check:
	$(PY) -m pytest tests/format \
		--cov=src/catan_llm/format \
		--cov-report=term-missing \
		--cov-fail-under=80
	$(PY) -m pytest tests/format \
		--cov=src/catan_llm \
		--cov-report=term-missing \
		--cov-fail-under=80

lint:
	$(PY) -m py_compile src/catan_llm/format/*.py

clean:
	rm -rf htmlcov coverage.xml .coverage .pytest_cache
