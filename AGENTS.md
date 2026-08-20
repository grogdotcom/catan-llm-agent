# Project Guidelines for AI Agents

## Backwards Compatibility

**We do not need backwards compatibility.**

When refactoring code or making changes, do not worry about maintaining backwards compatibility with previous versions. Focus on:
- Clean, maintainable code
- Proper separation of concerns
- Comprehensive test coverage
- Modern best practices

This allows for more aggressive refactoring and cleaner code architecture without being constrained by legacy requirements.

## Testing & Coverage

Run tests with `pytest` (see `pyproject.toml` for `pythonpath = ["src"]`):

```bash
venv/bin/python -m pytest -q          # all tests
venv/bin/python -m pytest tests/format -q  # format package only
```

Coverage is enforced in CI (`.github/workflows/ci.yml`) and locally via `make`:

```bash
make test              # pytest -q
make coverage          # html + xml reports
make coverage-check    # gate: --cov-fail-under=80 (mirrors CI)
open htmlcov/index.html
```

Config: `pyproject.toml` `[tool.coverage.*]` — `branch = true`, `source = ["src/catan_llm"]`, omits `collect_corpus.py` + deprecated shim, `fail_under = 80`. Current baseline: `src/catan_llm/format` ~88% branch / 90% line, `src/catan_llm` ~89%.