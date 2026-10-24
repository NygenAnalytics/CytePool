UVRUN ?= uv run --group docs

.PHONY: help html api execute check-outputs clean

help:
	@echo 'make html                              Build the API reference and the site'
	@echo 'make api                               Build only the API reference'
	@echo 'make execute PAGES="quickstart"        Execute notebooks and save their outputs'
	@echo 'make check-outputs PAGES="quickstart"  Check that saved outputs are still current'
	@echo 'make clean                             Remove docs/build'

api:
	@rm -rf build/api build/api-doctrees
	@$(UVRUN) sphinx-build -b html -n -W --keep-going -q -d build/api-doctrees \
		source/reference/api build/api

html: api
	@$(UVRUN) python build_docs.py

execute:
	$(if $(strip $(PAGES)),,$(error PAGES is required; name the notebooks to execute))
	@$(UVRUN) python execute_notebooks.py $(PAGES)

check-outputs:
	$(if $(strip $(PAGES)),,$(error PAGES is required; name the notebooks to check))
	@$(UVRUN) python execute_notebooks.py --check $(PAGES)

clean:
	rm -rf build
