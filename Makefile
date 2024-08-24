# Use bash for all commands
SHELL := /bin/bash

# Path to Conda base directory
CONDA_BASE := $(shell conda info --base)

# Env args
ENV_NAME=PGenv
REQUIREMENTS=requirements.txt


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
	. $(CONDA_BASE)/etc/profile.d/conda.sh && \
	conda create --name $(ENV_NAME) python=3.12 -y && \
	conda activate $(ENV_NAME) && \
	pip install -r requirements.txt && \
	echo "*************>>>>>>>>>>>>>>>>>>>>>>>>>    Please, make sure you have acces to HuggingFace Hub   <<<<<<<<<<<<<<********************" && \
	echo "*************>>>>>>>>>>>>>>>>>>>>>>>>>    If use have logged in to HF, ingnore me.              <<<<<<<<<<<<<<********************" && \
	echo "*************>>>>>>>>>>>>>>>>>>>>>>>>>    Else create .env file and set HF_TOKEN='HF_TOKEN'     <<<<<<<<<<<<<<********************"

check:
	. check.sh


run:
	. $(CONDA_BASE)/etc/profile.d/conda.sh  &&\
	conda activate $(ENV_NAME) &&\
	python inference.py \
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
	conda remove --name $(ENV_NAME) --all -y
	rm -rf $(MODEL_PATH)