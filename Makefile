ifndef DS_VOLUME
	DS_VOLUME=/scratch
endif

help:
	@echo "build -- builds the docker image"
	@echo "dockershell -- raises an interactive shell docker"

build:
	docker build -t iqfreg .
	./download.sh

dockershell:
	docker run --rm --name iqfreg --gpus all \
	-v $(shell pwd):/regressor -v $(DS_VOLUME):/scratch \
	-it iqfreg
