/*
 * Author: Sumit Bam Shrestha
 * 10/05/2019 11:00 AM
 * Contains routines that converts membrane potential of neuron into spikes
 */
#ifndef SPIKELOIHIKERNELS_H_INCLUDED
#define SPIKELOIHIKERNELS_H_INCLUDED

template <class T>
__global__ void getSpikesKernel(
	T* __restrict__ s,
	T* __restrict__ v,
	T* __restrict__ u,
	const T* __restrict__ weightedSpikes,
	const unsigned weightScale,
	const unsigned nNeurons, 
	const unsigned Ns, 
	const unsigned iDecay,	
	const unsigned vDecay,	
	const int theta)	// int because using unsigned value is giving errors when comparing the result with signed int
{
	unsigned neuronID = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(neuronID >= nNeurons)	return;
	
	int uOld, vOld;
	
	for(unsigned i=0; i<Ns; ++i)
	{
		unsigned linearID = i + neuronID * Ns;
		
		if(i==0)
		{
			uOld = weightScale * int(weightedSpikes[linearID]);
			vOld = uOld;
			
			u[linearID] = uOld;
			v[linearID] = uOld;
			s[linearID] = 0;
		}
		else
		{
			int uSign = (uOld >= 0) ? 1 : -1 ;
			int vSign = (vOld >= 0) ? 1 : -1 ;

			int uTemp = uSign * ( ( uSign * uOld * ( (1<<12) - iDecay ) ) >> 12 ) + weightScale * int(weightedSpikes[linearID]);
			int vTemp = vSign * ( ( vSign * vOld * ( (1<<12) - vDecay ) ) >> 12 ) + uTemp;
			
			s[linearID] = 0;
			u[linearID] = uOld = uTemp;

			if( vTemp > theta )
			{
				s[linearID] = 1;
				v[linearID] = vDecay;
				vOld = 0;
			}
			else
				v[linearID] = vOld = vTemp;
		}
	}
}


template <class T>
void getSpikes(
	T* __restrict__ s,
	T* __restrict__ v,
	T* __restrict__ u,
	const T* __restrict__ weightedSpikes,
	// const unsigned weightScale,
	const unsigned wgtExp,
	const unsigned nNeurons, 
	const unsigned Ns, 
	const unsigned iDecay,	
	const unsigned vDecay,	
	const unsigned theta)
{
	unsigned thread = 256;
	unsigned block  = ceil(1.0f * nNeurons / thread);
	// std::cout << "Ns : " << Ns << std::endl; 
	// std::cout << "iDecay : " << iDecay << std::endl; 
	// std::cout << "vDecay : " << vDecay << std::endl; 
	getSpikesKernel<T><<< block, thread >>>(s, v, u, weightedSpikes, 1 << (6 + wgtExp), nNeurons, Ns, iDecay, vDecay, theta);
}

#endif // SPIKELOIHIKERNELS_H_INCLUDED