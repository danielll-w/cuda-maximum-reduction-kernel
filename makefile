max_reduction: max_reduction.o
	nvcc -o max_reduction max_reduction.o -lm

max_reduction.o: max_reduction.cu 
	nvcc -c max_reduction.cu

clean:
	-rm *.o max_reduction 
