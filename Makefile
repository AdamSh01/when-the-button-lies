# When the Button Lies — convenience targets.
# First time:  pip install -e ".[dev]"   (add ,webworld to also reproduce the model run)

.PHONY: help install reproduce test figure clean

help:
	@echo "make install    - editable install with dev extras (matplotlib + pytest)"
	@echo "make reproduce  - regenerate results/calibration.png from results/raw/*.json"
	@echo "make test       - run the test suite (data invariants + wmeval logic tests)"
	@echo "make figure     - alias for reproduce"
	@echo "make clean      - remove the generated figure"

install:
	pip install -e ".[dev]"

reproduce figure:
	python make_figure.py

test:
	python -m pytest -q

clean:
	rm -f results/calibration.png
