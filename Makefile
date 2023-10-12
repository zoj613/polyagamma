.PHONY: clean cythonize install test sdist wheels

clean:
	rm -Rf build/* dist/* polyagamma/*.c polyagamma/*.so polyagamma/*.html \
		./**/polyagamma.egg-info **/*__pycache__ __pycache__ .coverage* \
		polyagamma/_version.py \

cythonize:
	cythonize polyagamma/*.pyx

dev:
	pip install -r requirements-dev.txt
	cythonize polyagamma/*.pyx
	pip install -e .
	pre-commit install --install-hooks

sdist: dev
	python -m build --sdist

wheel: dev
	python -m build --wheel

test: dev
	pytest tests/ -vvv

test-cov: clean
	cythonize polyagamma/*.pyx -X linetrace=True
	BUILD_WITH_COVERAGE=1 pip install -e .
	pytest -v --cov-branch --cov=polyagamma tests/ --cov-report=html
