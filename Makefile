.PHONY: test public-safety

test:
	python -m unittest discover -s tests

public-safety:
	python scripts/public_safety_scan.py
