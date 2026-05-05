.PHONY: uv tests quarto figures report wandb

FIGS := test_accuracy val_accuracy_curves cnn_ablation vit_ablation

BUILT_QMDS := $(addprefix build/figures/,$(addsuffix .qmd,$(FIGS)))
BUILT_SVGS  := $(addprefix figures/,$(addsuffix .svg,$(FIGS)))
SHARED_TPL  := src/templates/plot_figure.js.jinja src/templates/chunk.qmd.jinja

uv:
	uv lock && uv sync

tests:
	python -m unittest assignment_1_code/unittests/unittests.py

quarto:
	quarto add --no-prompt ruslan-basyrov/acuity

build/figures/%.qmd: src/templates/%.js.jinja $(SHARED_TPL)
	mkdir -p build/figures figures
	python src/scripts/template.py src/templates build/figures $* figures

figures/%.svg: build/figures/%.qmd
	deno run --allow-read --allow-write build/figures/svg_$*.js

figures: $(BUILT_QMDS) $(BUILT_SVGS)

report: figures
	quarto render index.qmd

wandb:
	python -m assignment_1_code.collect_results --out results/
