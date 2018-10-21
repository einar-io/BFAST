all:
  #nvcc main.c bfast-naive.cu  --compiler-options -std=c11  -o bfast 
	nvcc  bfast-naive.cu main.c -std=c++11 --compiler-options -std=c11  -o bfast

test: all
		../test.sh

clean:
	rm -f bfast
