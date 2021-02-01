.PHONY: clean cythonize install test sdist wheels

DOCKER_IMAGES=quay.io/pypa/manylinux1_x86_64 \
	      quay.io/pypa/manylinux2010_x86_64 \
	      quay.io/pypa/manylinux2014_x86_64

define make_wheels
	docker pull $(1)
	docker container run -t --rm -e PLAT=$(strip $(subst quay.io/pypa/,,$(1))) \
		-v $(shell pwd):/io $(1) /io/build_wheels.sh
endef


clean:
	rm -Rf build/* dist/* polyagamma/*.c polyagamma/*.so polyagamma/*.html \
		polyagamma.egg-info **/*__pycache__ __pycache__ .coverage* \
		wheelhouse/*

cythonize:
	cythonize polyagamma/*.pyx

install: clean cythonize
	poetry install

sdist: clean cythonize
	poetry build -f sdist

test:
	pytest tests/ -vvv

test-cov: clean
	poetry run cythonize polyagamma/*.pyx -X linetrace=True
	BUILD_WITH_COVERAGE=1 poetry install
	poetry run pytest -v --cov-branch --cov=polyagamma tests/ --cov-report=html
	
wheels: clean cythonize
	$(foreach img, $(DOCKER_IMAGES), $(call make_wheels, $(img));)
