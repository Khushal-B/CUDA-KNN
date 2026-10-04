NVCC ?= nvcc
ARCH ?= native
all: knn
knn: src/knn.cu
	$(NVCC) -O3 -arch=$(ARCH) -o knn src/knn.cu
clean:
	rm -f knn
