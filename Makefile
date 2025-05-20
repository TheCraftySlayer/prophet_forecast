all: features model metrics

features:
	python data_pipeline.py --config config.yaml --out features.csv

model:
	python pipeline.py config.yaml

metrics:
	@echo "Metrics exported to prophet_output"
