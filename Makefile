.PHONY: clean cythonize install test sdist

clean:
	rm -Rf build/* dist/* polyagamma/*.c polyagamma/*.so polyagamma/*.html \
		polyagamma.egg-info **/*__pycache__ __pycache__ .coverage*

cythonize:
	cythonize polyagamma/*.pyx

install: clean cythonize
	poetry install

sdist: clean cythonize
	poetry build -f sdist

test:
	pytest tests/ -vvv
