#include <torch/extension.h>
#include <vector>
#include "spikeLoihiKernels.h"

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_DEVICE(x, y) AT_ASSERTM(x.device().index() == y.device().index(), #x " and " #y " must be in same CUDA device")

// C++ Python interface

std::vector<torch::Tensor> getSpikesCuda(
	torch::Tensor weightedSpikes,
	// const unsigned weightScale,
	const unsigned wgtExp,
	const unsigned theta,
	const unsigned iDecay,
	const unsigned vDecay,
    const unsigned refDelay)
{
	CHECK_INPUT(weightedSpikes);

	auto current = torch::empty_like(weightedSpikes);
	auto voltage = torch::empty_like(weightedSpikes);
	auto spike   = torch::empty_like(weightedSpikes);

	// set the current cuda device to wherever the tensor d_u resides
	cudaSetDevice(weightedSpikes.device().index());
	
	unsigned Ns = weightedSpikes.size(-1);
	unsigned nNeurons = weightedSpikes.numel()/weightedSpikes.size(-1);
	
	// std::cout << "Ns = " << Ns << std::endl	
	// 		  << "nNeurons = " << nNeurons << std::endl;
	// std::cout << "refDelay = " << refDelay << std::endl;

	getSpikes<float>(spike.data<float>(),
					 voltage.data<float>(), 
					 current.data<float>(), 
					 weightedSpikes.data<float>(), 
					 // weightScale, nNeurons, Ns, iDecay, vDecay, theta);
					 wgtExp, nNeurons, Ns, iDecay, vDecay, refDelay, theta);

	// return {weightedSpikes, weightedSpikes, weightedSpikes};
	return {spike, voltage, current};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("getSpikes", &getSpikesCuda, "Get spikes for Loihi neuron (CUDA)");
}
