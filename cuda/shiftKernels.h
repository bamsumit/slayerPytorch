/*
 * Author: Sumit Bam Shrestha
 * 10/06/2019 6:30 PM
 * This header contains routines to perform tensor shifts as defined by the shift parameter
 */
#ifndef SHIFTKERNELS_H_INCLUDED
#define SHIFTKERNELS_H_INCLUDED

template <class T>
__global__ void shiftKernel(T* output,
							const T* input,
							const T shiftValue,
							unsigned signalSize, unsigned nNeurons, float Ts)
{
	// calcualte the threadID
	// this is the index of the signal along time axis
	unsigned tID = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned nID = blockIdx.y * blockDim.y + threadIdx.y;

	if(tID >= signalSize)	return;
	if(nID >= nNeurons)		return;

	// floor the shift to integer value
	int shiftBlocks = static_cast<int>(shiftValue/Ts);

	float temp = 0;
	auto neuronOffset = signalSize * nID;
	// shift the elements
	int id = tID - shiftBlocks;
	if(id >= 0 && id <signalSize)	temp = input[id + neuronOffset];

	output[tID + neuronOffset] = temp;
	return;
}

template <class T>
__global__ void shiftKernel(T* output,
							const T* input,
							const T* shiftLUT,
							unsigned signalSize, unsigned nNeurons, float Ts)
{
	// calcualte the threadID
	// this is the index of the signal along time axis
	unsigned tID = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned nID = blockIdx.y * blockDim.y + threadIdx.y;

	if(tID >= signalSize)	return;
	if(nID >= nNeurons)		return;

	// floor the shift to integer value
	int shiftBlocks = static_cast<int>(shiftLUT[nID]/Ts);

	float temp = 0;
	auto neuronOffset = signalSize * nID;
	// shift the elements
	int id = tID - shiftBlocks;
	if(id >= 0 && id <signalSize)	temp = input[id + neuronOffset];

	output[tID + neuronOffset] = temp;
	return;
}

template <class T>
void shift( T* output,
		    const T* input,
		    const T shiftValue,
		    unsigned signalSize, unsigned nNeurons, float Ts)
{
	dim3 thread(128, 8, 1);
	dim3 block( ceil( 1.0f * signalSize / thread.x ),
				ceil( 1.0f * nNeurons / thread.y ),
				1 );
	if(block.y >= 65535)	AT_ERROR("maximum blockDim.y exceeded.");
	if(block.z >= 65535)	AT_ERROR("maximum blockDim.z exceeded.");

	// std::cout << "Thread: (" << thread.x << ", " << thread.y << ", " << thread.z << ")" << std::endl;
	// std::cout << "Block : (" << block.x << ", " << block.y << ", " << block.z << ")" << std::endl;

	shiftKernel<T><<< block, thread >>>(output, 
										input, 
										shiftValue, 
										signalSize, nNeurons, Ts);

	// cudaDeviceSynchronize();
}

template <class T>
void shift( T* output,
		    const T* input,
		    const T* shiftLUT,
		    unsigned signalSize, unsigned nNeurons, unsigned nBatch, float Ts)
{
	dim3 thread(128, 8, 1);
	dim3 block( ceil( 1.0f * signalSize / thread.x ),
				ceil( 1.0f * nNeurons / thread.y ),
				1 );
	if(block.y >= 65535)	AT_ERROR("maximum blockDim.y exceeded.");
	if(block.z >= 65535)	AT_ERROR("maximum blockDim.z exceeded.");

	// std::cout << "Thread: (" << thread.x << ", " << thread.y << ", " << thread.z << ")" << std::endl;
	// std::cout << "Block : (" << block.x << ", " << block.y << ", " << block.z << ")" << std::endl;

	for(unsigned i=0; i<nBatch; ++i)
		shiftKernel<T><<< block, thread >>>(output + i * nNeurons * signalSize, 
											input + i * nNeurons * signalSize, 
											shiftLUT, 
											signalSize, nNeurons, Ts);

	// cudaDeviceSynchronize();
}


#endif // SHIFTKERNELS_H_INCLUDED