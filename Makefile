.PHONY: test public-safety release-audit

test:
	python -m unittest discover -s tests

public-safety:
	python scripts/public_safety_scan.py

release-audit:
	python scripts/release_audit.py
