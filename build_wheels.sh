#!/bin/bash
set -u -x
# adapted from pypa's python-manylinux-demo and
# https://github.com/pypa/python-manylinux-demo/blob/7e24ad2c202b6f58c55654f89c141bba749ca5d7/travis/build-wheels.sh

# navigate to the root of the mounted project
cd $(dirname $0)

bin_arr=(
    /opt/python/cp36-cp36m/bin
    /opt/python/cp37-cp37m/bin
    /opt/python/cp38-cp38/bin
    /opt/python/cp39-cp39/bin
)

# add  python to image's path
export PATH=/opt/python/cp37-cp37m/bin/:$PATH
# download install script
curl -#sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py > get-poetry.py
# install using local archive
python get-poetry.py -y --file poetry-1.1.4-linux.tar.gz


function build_poetry_wheels
{
    # build wheels for 3.6-3.9 with poetry 
    mkdir -p wheelhouse_temp
    for BIN in "${bin_arr[@]}"; do
        rm -Rf build/*
        # install build deps
        "${BIN}/python" ${HOME}/.poetry/bin/poetry run pip install numpy==1.19.0
        "${BIN}/python" ${HOME}/.poetry/bin/poetry build -f wheel
        for whl in dist/*.whl; do
            auditwheel repair "$whl" --plat $1 -w wheelhouse_temp
            "${BIN}/python" -m pip install wheelhouse_temp/*.whl
            # test if installed wheel imports correctly
            "${BIN}/python" -c \
                "from polyagamma import random_polyagamma; print(f'draw: {random_polyagamma()}');"
            mv wheelhouse_temp/*.whl wheelhouse/
            rm "$whl"
        done
    done
    rm -R wheelhouse_temp
}

build_poetry_wheels "$PLAT"
