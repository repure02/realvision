.PHONY: ci validate release-check docker-build

ci:
	python3 -m compileall -q src app
	python3 -m src.run_pipeline --help
	python3 -m src.inference.predict --help
	python3 -m src.utils.validate_dataset --allow_missing
	python3 -m src.utils.validate_final_artifacts --allow_missing
	@if [ -f reports/logo_summary.csv ]; then python3 -m src.utils.validate_logo_baseline --allow_missing_details; else echo "reports/logo_summary.csv not found; skipping LOGO baseline validation."; fi

validate:
	python3 -m src.utils.validate_dataset
	python3 -m src.utils.validate_final_artifacts
	python3 -m src.utils.validate_logo_baseline

release-check:
	test -f checkpoints/convnext_tiny_final_inference_best.pt
	python3 -m src.utils.validate_dataset
	python3 -m src.utils.validate_final_artifacts
	python3 -m src.utils.validate_logo_baseline

docker-build:
	docker build -t realvision:local .
