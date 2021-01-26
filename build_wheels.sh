#!/bin/bash
set -e -u -x
# adapted from pypa's python-manylinux-demo and
# https://github.com/pypa/python-manylinux-demo/blob/7e24ad2c202b6f58c55654f89c141bba749ca5d7/travis/build-wheels.sh

# navigate to the root of the mounted project
cd $(dirname $0)

bin_arr=(
    /opt/python/cp36-cp36m/bin
    /opt/python/cp37-cp37m/bin
    /opt/python/cp39-cp39/bin
    /opt/python/cp38-cp38/bin
)

# add  python to image's path
export PATH=/opt/python/cp38-cp38/bin/:$PATH
# download install script
curl -#sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py > get-poetry.py
# install using local archive
python get-poetry.py -y --file poetry-1.1.4-linux.tar.gz


function build_poetry_wheels
{
    # build wheels for 3.6-3.9 with poetry 
    for BIN in "${bin_arr[@]}"; do
        rm -Rf build/*
        # install build deps
        "${BIN}/python" ${HOME}/.poetry/bin/poetry run pip install numpy
        "${BIN}/python" ${HOME}/.poetry/bin/poetry build -f wheel
        auditwheel repair dist/*.whl --plat $1
        whl="$(basename dist/*.whl)"
        "${BIN}/python" -m pip install wheelhouse/"$whl"
        # test if installed wheel imports correctly
        "${BIN}/python" -c \
            "from polyagamma import polyagamma; print(f'draw: {polyagamma()}');"
        rm dist/*.whl
    done
}

build_poetry_wheels "$PLAT"
