__global__ void warmup_kernel(int* c, int n) {
	int sum = 0;

#pragma unroll 1
	for (int i = 0; i < n; i++) {
		sum++;
	}
	c[0] = sum;
}


void warmupKernel(cudaStream_t stream, int *c, int n)
{
	warmup_kernel << <1, 1, 0, stream >> > (c, n);
}

void checkWarmUp(cudaStream_t stream, int warmUpMs)
{
	if (warmUpMs > 0)
	{
		cudaEvent_t start, end;
		int *warmupBuffer = nullptr;

		unsigned int cudaEventFlags = cudaEventDefault;// : cudaEventBlockingSync;
		cudaEventCreateWithFlags(&start, cudaEventFlags);
		cudaEventCreateWithFlags(&end, cudaEventFlags);

		cudaMalloc((void**)&warmupBuffer, sizeof(int));
		float totalMs = 0;

		do
		{
			cudaEventRecord(start, stream);
			warmupKernel(stream, warmupBuffer, 1 << 20);
			cudaEventRecord(end, stream);
			cudaEventSynchronize(end);
			float ms;
			cudaEventElapsedTime(&ms, start, end);
			totalMs += ms;
		} while ((int)totalMs < warmUpMs);

		cudaEventDestroy(start);
		cudaEventDestroy(end);
		cudaFree(warmupBuffer);
	}
}