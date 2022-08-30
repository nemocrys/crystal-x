# USAGE INSTSRUCTIONS
# 
# Build image (run this in the main directory (where .git folder is)):
# docker build -t nemocrys/dolfinx:2022.08.16 -f ./docker/Dockerfile .
#
# Run image:
# docker run --rm -it --mount type=bind,source="$(pwd)",target=/root nemocrys/dolfinx:2022.08.16


FROM dolfinx/dolfinx:v0.5.0

RUN pip3 install \
    meshio==4.4.*\
    pyyaml==5.4.*\
    objectgmsh==0.*\