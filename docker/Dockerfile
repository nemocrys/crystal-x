# USAGE INSTSRUCTIONS
# 
# Build image (run this in the main directory (where .git folder is)):
# docker build -t nemocrys/dolfinx:v0.5.2 -f ./docker/Dockerfile .
#
# Run image and map current directory to home/workdir on Linux:
# docker run -it --rm -v $PWD:/home/workdir -e LOCAL_UID=$(id -u $USER) -e LOCAL_GID=$(id -g $USER) nemocrys/dolfinx:v0.5.2 bash
# and on Windows:
# docker run -it --rm -v ${PWD}:/home/workdir nemocrys/dolfinx:v0.5.2 bash


FROM dolfinx/dolfinx:v0.5.2

RUN pip3 install \
    meshio==4.4.*\
    pyyaml==5.4.*\
    objectgmsh==0.*

RUN mkdir /home/workdir
WORKDIR /home/workdir

# modify user id and group
# see https://techflare.blog/permission-problems-in-bind-mount-in-docker-volume/
RUN apt-get update && \
    apt-get -y install gosu
COPY  ./docker/entrypoint.sh /home/entrypoint.sh
ENTRYPOINT [ "/home/entrypoint.sh" ]
