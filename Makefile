.PHONY: install seed servers pipeline app test

install:
	pip install -r requirements.txt

seed:
	python scripts/seed_db.py

servers:
	bash scripts/run_servers.sh

pipeline:
	python scripts/run_pipeline.py

app:
	streamlit run src/marketmind/app/streamlit_app.py

test:
	python scripts/test_quant.py
