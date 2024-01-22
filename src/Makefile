kernels = kernels/bfast_others.cu  \
	  kernels/bfast_step_2.cu  \
	  kernels/bfast_step_4a.cu \
	  kernels/bfast_step_4c.cu \
	  kernels/bfast_step_6.cu  \
	  kernels/bfast_step_8.cu

all: 	datasets test benchmark-naive benchmark-alt benchmark

build:
	gcc -std=c11 -c values.c
	gcc -std=c11 -c main.c
	nvcc -Xptxas -O3,-v -use_fast_math $(kernels) bfast.cu bfast_util.cu main.o values.o -std=c++11 -o bfast

debug:
	gcc -g -std=c11 -c values.c
	gcc -g -std=c11 -c main.c
	nvcc -g -G $(kernels) bfast.cu bfast_util.cu main.o values.o -std=c++11 -o bfast

test: build
	time futhark test --runner=./bfast tests/tests.fut

test-memcheck: build
	time futhark test --runner=./memcheck-runner.sh tests/tests.fut

benchmark: build
	time gunzip -c ../data/sahara.in.gz | ./bfast -e bfast-opt -i 1 -r 50 2>&1 >/dev/null

benchmark-alt: build
	time gunzip -c ../data/sahara.in.gz | ./bfast -e bfast-opt-alt -i 1 -r 50 2>&1 >/dev/null

benchmark-naive: build
	time gunzip -c ../data/sahara.in.gz | ./bfast -e bfast-naive -i 1 -r 50 2>&1 >/dev/null

datasets:
	fut/generate-datasets.sh

clean:
	rm -f bfast main.o values.o
