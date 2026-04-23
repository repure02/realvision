.PHONY: ci validate docker-build

ci:
	python3 -m compileall -q src app
	python3 -m src.run_pipeline --help
	python3 -m src.inference.predict --help
	python3 -m src.utils.validate_dataset --allow_missing
	python3 -m src.utils.validate_final_artifacts --allow_missing
	@if [ -f reports/logo_summary.csv ]; then python3 -m src.utils.validate_logo_baseline; else echo "LOGO summary not found; skipping LOGO baseline validation."; fi

validate:
	python3 -m src.utils.validate_dataset
	python3 -m src.utils.validate_final_artifacts
	python3 -m src.utils.validate_logo_baseline

docker-build:
	docker build -t realvision:local .
