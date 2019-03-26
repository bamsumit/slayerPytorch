	/*
 * Author: Sumit Bam Shrestha
 * 12/03/2018 12:00 PM
 * Contains routines that converts membrane potential of neuron into spikes
 */

// Modified version including pytorch interfacing logic
 
#ifndef SPIKEKERNELS_H_INCLUDED
#define SPIKEKERNELS_H_INCLUDED

#include <torch/torch.h>

__global__ void getSpikesKernel(float* __restrict__ d_s, float* __restrict__ d_u, const float* __restrict__ d_nu, unsigned nBatch, \
								unsigned nNeurons, unsigned nuSize, unsigned batchStride, unsigned Ns, float theta, float Ts)
{
	unsigned batchID  = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned neuronID = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned startID  = batchID * batchStride;
	const float spike = 1.0f/Ts;
	
	if(batchID  >= nBatch)				return;
	if(neuronID >= nNeurons)	return;
	
	unsigned thread = 32;
	unsigned block  = ceil(1.0f * nuSize / thread);
		
	for(unsigned i=0; i<batchStride; ++i)
	{
		unsigned linearID = startID + i + neuronID * Ns;
		if(d_u[linearID] >= theta)
		{
			d_s[linearID] = spike;
			// dynamic parallelism seems to be slower because of race condition!!!
			// ahpKernel<<< block, thread >>>(d_u + linearID, d_nu, nuSize);
			// cudaDeviceSynchronize();
			for(unsigned j=0; j<nuSize; ++j)
			{
				if(startID + i + j < Ns)	d_u[linearID + j] += d_nu[j];
			}
		}
		else	d_s[linearID] = 0.0f;
	}
}

void getSpikes(float* d_s, float* d_u, const float* d_nu, unsigned nNeurons, unsigned nuSize, unsigned Ns, unsigned batchStride, float theta, float Ts)
{
	unsigned nBatch = Ns/batchStride;
	dim3 thread, block;
	thread.x = 32;
	thread.y = 8;
	block.x  = ceil(1.0f * nBatch   / thread.x);
	block.y  = ceil(1.0f * nNeurons / thread.y);
	getSpikesKernel<<< block, thread >>>(d_s, d_u, d_nu, nBatch, nNeurons, nuSize, batchStride, Ns, theta, Ts);
}

__global__ void evalRhoKernel(float* d_rho, const float* d_u, float theta, float tau, unsigned nNeurons, unsigned Ns)
{
	unsigned timeID = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned nID    = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(timeID >= Ns || nID >= nNeurons)	return;
	
	unsigned linearID = timeID + nID * Ns;
	
	d_rho[linearID] = 1/tau * exp(-fabs(theta - d_u[linearID])/tau);
}

std::vector<at::Tensor> getSpikesCuda(at::Tensor d_u, at::Tensor d_s, const at::Tensor& d_nu, const float theta, const float Ts)
{
	unsigned nuSize = d_nu.size(-1);
	unsigned Ns = d_u.size(-1);
	// Parallelization is done in neuron dimension, run multiple batches through neuron duplication
	unsigned nNeurons = d_u.size(0) * d_u.size(1) * d_u.size(2) * d_u.size(3);
	unsigned batchStride = Ns;
	getSpikes(d_s.data<float>(), d_u.data<float>(), d_nu.data<float>(), nNeurons, nuSize, Ns, batchStride, theta, Ts);
	return {d_u, d_s};
}

#endif // SPIKEKERNELS_H_INCLUDED