all: opt

naive:
	nvcc  bfast-naive.cu main.c -std=c++11 --compiler-options -std=c11  -o bfast

opt:
	nvcc  bfast-opt.cu main.c -std=c++11 --compiler-options -std=c11  -o bfast

test: all
		../test.sh

clean:
	rm -f bfast
