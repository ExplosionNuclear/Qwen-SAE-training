IMAGE_NAME := akkadeeemikk/mats_sae_training
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

sheduled_sae:
	INSTANCE_ID=$$(echo $$(cat ~/.vast_containerlabel) | grep -o '[0-9]\+'); \
	trap "vastai stop instance $$INSTANCE_ID" EXIT; \
	make train_sae

train_sae:
	python train_sae.py 0 6 
	python train_sae.py 7 12 
	python train_sae.py 13 18 
	python train_sae.py 19 23 
	python train_sae.py 24 27 

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
