IMAGE_NAME := akkadeeemikk/mats
CONTAINER_NAME := mats_sae

build_mats:
	docker build -f docker/Dockerfile -t $(IMAGE_NAME) .

stop:
	docker stop $(CONTAINER_NAME)

jupyter:
	jupyter lab --allow-root --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token=mats

run_docker:
	docker run -d -it --rm \
		--ipc=host \
		--network=host \
		--gpus=all \
		-v ./:/workspace/ \
		-v ./.cache/huggingface:/root/.cache/huggingface \
		--name $(CONTAINER_NAME) \
		$(IMAGE_NAME) bash

enter:
	docker exec -it $(CONTAINER_NAME) bash

format:
	poetry run black .
	poetry run isort .

check-format:
	poetry run flake8 .
	poetry run black --check .
	poetry run isort --check-only --diff .

check-type:
	poetry run pyright .

test:
	make unit-test
	make acceptance-test

unit-test:
	poetry run pytest -v --cov=sae_training/ --cov-report=term-missing --cov-branch tests/unit

acceptance-test:
	poetry run pytest -v --cov=sae_training/ --cov-report=term-missing --cov-branch tests/acceptance
