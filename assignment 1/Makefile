uv:
	uv lock && uv sync

tests:
	python -m unittest assignment_1_code/unittests/unittests.py

quarto:
	quarto add --no-prompt ruslan-basyrov/acuity

report:
	quarto render index.qmd
