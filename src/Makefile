all: opt

naive:
	nvcc  bfast-naive.cu main.c -std=c++11 --compiler-options -std=c11  -o bfast

opt:
	nvcc  bfast-opt.cu main.c -std=c++11 --compiler-options -std=c11  -o bfast

test: all
	futhark-test --compiler=/bin/true --runner=./bfast tests/tests.fut

test-memcheck: all
	futhark-test --compiler=/bin/true --runner=./memcheck-runner.sh tests/tests.fut

benchmark: all
	gunzip -c ../data/sahara.in.gz | ./bfast -e bfast -i 1 -r 50 2>&1 >/dev/null


clean:
	rm -f bfast
