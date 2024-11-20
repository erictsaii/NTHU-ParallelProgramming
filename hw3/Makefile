NVFLAGS  := -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 
LDFLAGS  := -lm
EXES     := hw3-1 hw3-2 hw3-3

alls: $(EXES)

clean:
	rm -f $(EXES)

hw3-1: hw3-1.cc
	g++ -O3 -pthread    hw3-1.cc   -o hw3-1

hw3-2: hw3-2.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?

hw3-3: hw3-3.cu
	nvcc $(NVFLAGS) -Xcompiler -fopenmp $(LDFLAGS) -o $@ $?
