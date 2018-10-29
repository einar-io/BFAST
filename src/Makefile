
kernels = kernels/bfast_others.cu kernels/bfast_step_2.cu \
					kernels/bfast_step_4a.cu kernels/bfast_step_4c.cu \
					kernels/bfast_step_6.cu kernels/bfast_step_8.cu

all:
	gcc -std=c11 -c values.c
	gcc -std=c11 -c main.c
	nvcc $(kernels) bfast.cu bfast_util.cu main.o values.o -std=c++11 -o bfast

debug:
	gcc -g -std=c11 -c values.c
	gcc -g -std=c11 -c main.c
	nvcc -g -G $(kernels) bfast.cu bfast_util.cu main.o values.o -std=c++11 -o bfast

test: all
	time futhark-test --compiler=/bin/true --runner=./bfast tests/tests.fut

test-memcheck: all
	time futhark-test --compiler=/bin/true --runner=./memcheck-runner.sh tests/tests.fut

benchmark: all
	time gunzip -c ../data/sahara.in.gz | ./bfast -e bfast-opt -i 1 -r 50 2>&1 >/dev/null

benchmark-naive: all
	time gunzip -c ../data/sahara.in.gz | ./bfast -e bfast-naive -i 1 -r 50 2>&1 >/dev/null

clean:
	rm -f bfast main.o values.o

