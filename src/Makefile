all: opt

naive:
	nvcc  bfast-naive.cu main.c -std=c++11 --compiler-options -std=c11  -o bfast

opt:
	nvcc  bfast-opt.cu main.c -std=c++11 --compiler-options -std=c11  -o bfast

test: all
	futhark-test --compiler=/bin/true --runner=./bfast tests/tests.fut

test-memcheck: all
	futhark-test --compiler=/bin/true --runner=./memcheck-runner.sh tests/tests.fut

clean:
	rm -f bfast
