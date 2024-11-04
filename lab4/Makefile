NVCC     = nvcc 
NVCCFLAGS  = -O3 -std=c++11 -Xptxas=-v -arch=sm_61 --use_fast_math

LDFLAGS = -lpng -lz

TARGETS = sobel sobel-amd

sobel: sobel.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ $<

.PHONY: clean
clean:
	rm -f $(TARGETS)
