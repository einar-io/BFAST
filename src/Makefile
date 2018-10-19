all:
	nvcc bfast-naive.cu  main.c --compiler-options -std=c11  -o bfast 

test: all
		./test.sh

clean:
	rm -f bfast
