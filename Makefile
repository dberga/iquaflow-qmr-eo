ifndef DS_VOLUME
	DS_VOLUME=/scratch
endif

help:
	@echo "build -- builds the docker image"
	@echo "dockershell -- raises an interactive shell docker"

build:
	docker build -t sisr-reg .
	./download.sh

dockershell:
	docker run --rm --name sisr-reg --gpus all -p 9193:9193 \
	-v $(shell pwd):/sisr -v $(DS_VOLUME):/scratch \
	-it sisr-reg
