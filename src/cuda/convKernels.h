/*
 * Author: Sumit Bam Shrestha
 * 09/05/2019 6:00 PM
 * This header contains routines to perform time based convolution and correlation of signal
 * These operations are key in forward propagation and backpropagation routines in SLAYER
 */
#ifndef CONVKERNELS_H_INCLUDED
#define CONVKERNELS_H_INCLUDED

template <class T>
__global__ void convKernel(	T* output, 
							const T* input, const T* filter, 
							unsigned signalSize, unsigned filterSize, unsigned nNeurons,
							float Ts)
{
	// calcualte the threadID
	// this is the index of the signal along time axis
	unsigned tID = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned nID = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(tID >= signalSize)	return;
	if(nID >= nNeurons)		return;
	
	// declare local variables
	float result = 0.0f;
	
	// calculate convolution sum
	for(unsigned i=0; i<filterSize; ++i)
	{
		int id = tID - i;
		if(id >= 0)		result += input[id + nID * signalSize] * filter[i];
	}
	output[tID + nID * signalSize] = result * Ts;	
	return;
}

template <class T>
__global__ void corrKernel(	T* output, 
							const T* input, const T* filter, 
							unsigned signalSize, unsigned filterSize, unsigned nNeurons,
							float Ts)
{
	// calcualte the threadID
	// this is the index of the signal along time axis
	unsigned tID = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned nID = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(tID >= signalSize)	return;
	if(nID >= nNeurons)		return;
	
	// declare local variables
	float result = 0.0f;
	
	// calculate convolution sum
	for(unsigned i=0; i<filterSize; ++i)
	{
		int id = tID + i;
		if(id < signalSize)		result += input[id + nID * signalSize] * filter[i];
	}
	output[tID + nID * signalSize] = result * Ts;
	return;
}

template <class T>
void conv(	T* output, 
			const T* input, const T* filter, 
			unsigned signalSize, unsigned filterSize, unsigned nNeurons, 
			float Ts)
{
	dim3 thread(128, 8, 1);

	// int nGrid = 128;
	int nGrid = ceil( 1.0f * nNeurons / thread.y / 65535);
	int neuronsPerGrid = ceil(1.0f * nNeurons / nGrid);

	for(auto i=0; i<nGrid; ++i)
	{
		int startOffset = i * neuronsPerGrid;
		int neuronsInGrid = (startOffset + neuronsPerGrid <= nNeurons) ? neuronsPerGrid : nNeurons - startOffset;

		if(neuronsInGrid < 0)	break;

		dim3 block(	ceil( 1.0f * signalSize    /thread.x ), 
					ceil( 1.0f * neuronsInGrid /thread.y ), 
					1 );

		// these should never be trigerred
		if(block.y >= 65535)	AT_ERROR("maximum blockDim.y exceeded.");
		if(block.z >= 65535)	AT_ERROR("maximum blockDim.z exceeded.");

		convKernel<T><<< block, thread >>>( output + startOffset * signalSize, 
											input  + startOffset * signalSize, 
											filter, signalSize, filterSize, 
											neuronsInGrid, Ts);
	}
}

template <class T>
void corr(	T* output, 
			const T* input, const T* filter, 
			unsigned signalSize, unsigned filterSize, unsigned nNeurons,
			float Ts)
{
	dim3 thread(128, 8, 1);

	// int nGrid = 128;
	int nGrid = ceil( 1.0f * nNeurons / thread.y / 65535 );
	int neuronsPerGrid = ceil(1.0f * nNeurons / nGrid);
	for(auto i=0; i<nGrid; ++i)
	{
		int startOffset = i * neuronsPerGrid;
		int neuronsInGrid = (startOffset + neuronsPerGrid <= nNeurons) ? neuronsPerGrid : nNeurons - startOffset;

		if(neuronsInGrid < 0)	break;

		dim3 block(	ceil( 1.0f * signalSize    /thread.x ), 
					ceil( 1.0f * neuronsInGrid /thread.y ), 
					1 );

		// these should never be trigerred
		if(block.y >= 65535)	AT_ERROR("maximum blockDim.y exceeded.");
		if(block.z >= 65535)	AT_ERROR("maximum blockDim.z exceeded.");

		corrKernel<T><<< block, thread >>>( output + startOffset * signalSize, 
											input  + startOffset * signalSize, 
											filter, signalSize, filterSize, 
											neuronsInGrid, Ts);
	}
}

#endif // CONVKERNELS_H_INCLUDED
