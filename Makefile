# MarketMind — clone-and-run targets.
#
#   make install   create .venv and install dependencies
#   make seed      seed the SQLite portfolio
#   make servers   start the 3 MCP servers (Ctrl-C to stop)
#   make app       launch the Streamlit dashboard
#   make demo      seed -> servers (background) -> app (the MVP launch path)
#
# Fresh checkout:  make install && make seed && make demo

VENV := .venv

# Use the venv interpreter on both POSIX (.venv/bin) and Windows (.venv/Scripts).
ifeq ($(OS),Windows_NT)
  PY := $(VENV)/Scripts/python.exe
else
  PY := $(VENV)/bin/python
endif

.PHONY: install seed servers app demo pipeline test clean

install:
	python -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	@echo
	@echo "Installed. Next: cp .env.example .env  (add your free GROQ_API_KEY)"

seed:
	$(PY) scripts/seed_db.py

servers:
	PYTHON=$(PY) bash scripts/run_servers.sh

app:
	PYTHONPATH=src $(PY) -m streamlit run src/marketmind/app/streamlit_app.py

# seed -> start servers in the background -> run the app -> stop servers on exit.
demo: seed
	@bash -c '\
	  PYTHON=$(PY) bash scripts/run_servers.sh & \
	  SERVERS_PID=$$!; \
	  trap "kill $$SERVERS_PID 2>/dev/null" EXIT INT TERM; \
	  PYTHONPATH=src $(PY) -m streamlit run src/marketmind/app/streamlit_app.py'

pipeline:
	$(PY) scripts/run_pipeline.py

test:
	$(PY) scripts/test_agents.py

clean:
	rm -rf $(VENV)
