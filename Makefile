PROJ_NAME=eoqmr
CONTAINER_NAME="${PROJ_NAME}-${USER}"

ifndef DS_VOLUME
	DS_VOLUME=/scratch
endif

ifndef NB_PORT
	NB_PORT=8889
endif

ifndef MLF_PORT
	MLF_PORT=5000
endif

ifndef SHM_SIZE
	SHM_SIZE="8g"
endif

help:
	@echo "build -- builds the docker image"
	@echo "dockershell -- raises an interactive shell docker"
	@echo "notebookshell -- launches a notebook server"
	@echo "mlflow -- launches an mlflow server"

build:
	docker build -t $(PROJ_NAME) .

download:
	chmod 775 ./download.sh
	./download.sh

dockershell:
	docker run --rm --name $(CONTAINER_NAME) --gpus all -p 9199:9199 \
	-v $(shell pwd):/$(PROJ_NAME) -v $(DS_VOLUME):/scratch -v \
	-it $(PROJ_NAME)

notebookshell:
	docker run --gpus all --privileged -itd --name $(CONTAINER_NAME)-nb \
	-p ${NB_PORT}:${NB_PORT} \
	-v $(shell pwd):/$(PROJ_NAME) -v $(DS_VOLUME):/scratch \
	$(PROJ_NAME) \
	jupyter notebook \
	--NotebookApp.token=$(PROJ_NAME) \
	--no-browser \
	--ip=0.0.0.0 \
	--allow-root \
	--port=${NB_PORT} \
	--shm-size ${SHM_SIZE}

mlflow:
	docker run --privileged -itd --rm --name $(CONTAINER_NAME)-mlf \
	-p ${MLF_PORT}:${MLF_PORT} \
	-v $(shell pwd):/$(PROJ_NAME) -v $(DS_VOLUME):/scratch \
	$(PROJ_NAME) \
	mlflow ui --host 0.0.0.0:${MLF_PORT}
