.PHONY: install test check-data eval study app docker

install:
	pip install -r requirements-dev.txt

test:
	python -m pytest -q

check-data:
	python -m rag_assistant check-data

eval:            # regression gate used by CI (in-sample; see `study` for the honest estimate)
	python -m rag_assistant eval --gate

study:           # nested cross-validation + ablations + full report in reports/latest
	python -m rag_assistant study

app:
	streamlit run app.py

docker:
	docker build -t rag-assistant .
