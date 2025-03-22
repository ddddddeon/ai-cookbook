.PHONY:
run: venv
	python main.py

.PHONY:
seed: venv
	python main.py -s

.PHONY:
install: venv
	uv sync

.PHONY:
check: venv
	mypy *.py


.PHONY:
fmt:
	ruff format
