.PHONY: uv tests quarto figures report wandb

FIGS := test_accuracy

uv:
	uv lock && uv sync

tests:
	python -m unittest assignment_1_code/unittests/unittests.py

quarto:
	quarto add --no-prompt ruslan-basyrov/acuity

figures:
	mkdir -p build/figures figures
	$(foreach fig,$(FIGS), \
		python src/scripts/template.py src/templates build/figures $(fig) figures && \
		deno run --allow-read --allow-write build/figures/svg_$(fig).js ;)

report: figures
	quarto render index.qmd

wandb:
	python -m assignment_1_code.collect_results --out results/
