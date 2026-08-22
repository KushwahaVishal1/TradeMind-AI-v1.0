.PHONY: install test protected lint fmt clean init dashboard docker ci

install:
	pip install -e ".[dev]"

init:          ## create the operational store
	python main.py init

protected:     ## the invariants that block a merge — seconds, not minutes
	PYTHONHASHSEED=42 pytest -m protected -v

test:          ## full suite
	PYTHONHASHSEED=42 pytest -q

lint:
	ruff check src tests dashboard

fmt:
	ruff check --fix src tests
	ruff format src tests

dashboard:     ## launch the Streamlit dashboard
	streamlit run dashboard/app.py

docker:
	docker build -t trademind-ai:latest .

ci:            ## everything CI runs, locally
	$(MAKE) lint
	$(MAKE) protected
	$(MAKE) test

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
