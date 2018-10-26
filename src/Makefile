all: naive

bfast:
	nvcc bfast-naive.cu bfast_util.cu main.c values.c -std=c++11 --compiler-options -std=c11  -o bfast

bfast-debug:
	nvcc -g -G --generate-line-info bfast-naive.cu bfast_util.cu main.c values.c -std=c++11 --compiler-options -std=c11  -o bfast

test: all
	time futhark-test --compiler=/bin/true --runner=./bfast tests/tests.fut

test-memcheck: all
	time futhark-test --compiler=/bin/true --runner=./memcheck-runner.sh tests/tests.fut

benchmark: all
	time gunzip -c ../data/sahara.in.gz | ./bfast -e bfast -i 1 -r 50 2>&1 >/dev/null

clean:
	rm -f bfast
