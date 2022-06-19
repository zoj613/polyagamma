.PHONY: clean cythonize install test sdist wheels

clean:
	rm -Rf build/* dist/* polyagamma/*.c polyagamma/*.so polyagamma/*.html \
		./**/polyagamma.egg-info **/*__pycache__ __pycache__ .coverage* \

cythonize:
	cythonize polyagamma/*.pyx

dev:
	pip install -r requirements-dev.txt

sdist:
	python -m build --sdist

wheel:
	python -m build --wheel

test: cythonize
	pytest tests/ -vvv

test-cov: clean
	cythonize polyagamma/*.pyx -X linetrace=True
	BUILD_WITH_COVERAGE=1 pip install -e .
	pytest -v --cov-branch --cov=polyagamma tests/ --cov-report=html
