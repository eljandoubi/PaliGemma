# Use bash for all commands
SHELL := /bin/bash

# Inference args
MODEL_ID=google/paligemma-3b-pt-224
MODEL_PATH=$(HOME)/paligemma-weights
PROMPT=this tower is 
IMAGE_FILE_PATH=samples/EiffelTower.jpg
MAX_TOKENS_TO_GENERATE=100
TEMPERATURE=0.8
TOP_P=0.9
DO_SAMPLE=False
ONLY_CPU=False

build:
	uv sync && \
	echo "*************>>>>>>>>>>>>>>>>>>>>>>>>>    Please, make sure you have access to HuggingFace Hub   <<<<<<<<<<<<<<********************" && \
	echo "*************>>>>>>>>>>>>>>>>>>>>>>>>>    If you are already logged in to HF, ignore me.         <<<<<<<<<<<<<<********************" && \
	echo "*************>>>>>>>>>>>>>>>>>>>>>>>>>    Else create .env file and set HF_TOKEN='HF_TOKEN'      <<<<<<<<<<<<<<********************"

check:
	. check.sh

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

run:
	uv run python inference.py \
		--model-id "$(MODEL_ID)" \
		--model-path "$(MODEL_PATH)" \
		--prompt "$(PROMPT)" \
		--image-file-path "$(IMAGE_FILE_PATH)" \
		--max-tokens-to-generate $(MAX_TOKENS_TO_GENERATE) \
		--temperature $(TEMPERATURE) \
		--top-p $(TOP_P) \
		--do-sample "$(DO_SAMPLE)" \
		--only-cpu "$(ONLY_CPU)"

clean:
	rm -rf .venv $(MODEL_PATH)