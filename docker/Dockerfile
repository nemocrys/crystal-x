# USAGE INSTSRUCTIONS
# 
# Build image (run this in the main directory (where .git folder is)):
# docker build -t nemocrys/dolfinx:2022.07.27 -f ./docker/Dockerfile .
#
# Run image:
# docker run --rm -it --mount type=bind,source="$(pwd)",target=/root nemocrys/dolfinx:2022.07.27


FROM dolfinx/dolfinx:latest

RUN pip3 install \
    meshio==4.4.*\
    pyyaml==5.4.*\
    objectgmsh==0.*\