#include <torch/extension.h>
#include <vector>
#include "spikeKernels.h"
#include "convKernels.h"
#include "shiftKernels.h"

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_DEVICE(x, y) AT_ASSERTM(x.device().index() == y.device().index(), #x " and " #y " must be in same CUDA device")

// C++ Python interface

torch::Tensor getSpikesCuda(
	torch::Tensor d_u,
	const torch::Tensor& d_nu,
	const float theta,
	const float Ts)
{
	CHECK_INPUT(d_u);
	CHECK_INPUT(d_nu);

	// check if tensor are in same device
	CHECK_DEVICE(d_u, d_nu);

	auto d_s = torch::empty_like(d_u);

	// TODO implement for different data types

	// set the current cuda device to wherever the tensor d_u resides
	cudaSetDevice(d_u.device().index());

	unsigned nuSize = d_nu.size(-1);
	unsigned Ns = d_u.size(-1);
	unsigned nNeurons = d_u.size(0) * d_u.size(1) * d_u.size(2) * d_u.size(3);
	getSpikes<float>(d_s.data<float>(), d_u.data<float>(), d_nu.data<float>(), nNeurons, nuSize, Ns, theta, Ts);

	return d_s;
}

torch::Tensor convCuda(torch::Tensor input, torch::Tensor filter, float Ts)
{
	CHECK_INPUT(input);
	CHECK_INPUT(filter);
	CHECK_DEVICE(input, filter);

	cudaSetDevice(input.device().index());

	auto output = torch::empty_like(input);
	
	unsigned signalSize = input.size(-1); 
	unsigned filterSize = filter.numel();
	unsigned nNeurons   = input.numel()/input.size(-1); 
	conv<float>(output.data<float>(), input.data<float>(), filter.data<float>(), signalSize, filterSize, nNeurons, Ts);

	return output;
}

torch::Tensor corrCuda(torch::Tensor input, torch::Tensor filter, float Ts)
{
	CHECK_INPUT(input);
	CHECK_INPUT(filter);
	CHECK_DEVICE(input, filter);

	cudaSetDevice(input.device().index());

	auto output = torch::empty_like(input);
	
	unsigned signalSize = input.size(-1); 
	unsigned filterSize = filter.numel();
	unsigned nNeurons   = input.numel()/input.size(-1); 
	corr<float>(output.data<float>(), input.data<float>(), filter.data<float>(), signalSize, filterSize, nNeurons, Ts);

	return output;
}

torch::Tensor shiftCuda(torch::Tensor input, torch::Tensor shiftLUT, float Ts)
{
	CHECK_INPUT(input);
	CHECK_INPUT(shiftLUT);
	CHECK_DEVICE(input, shiftLUT);

	cudaSetDevice(input.device().index());

	auto output = torch::empty_like(input);

	if(shiftLUT.numel() == 1)
	{
		unsigned signalSize = input.size(-1);
		unsigned nNeurons   = input.numel()/signalSize;
		
		float shiftValue = shiftLUT.item<float>();
		
		shift<float>(output.data<float>(), input.data<float>(), shiftValue, signalSize, nNeurons, Ts);		
	}
	else
	{
		unsigned signalSize = input.size(-1);
		unsigned nBatch     = input.size(0);
		unsigned nNeurons   = input.numel()/signalSize/nBatch;
		
		AT_ASSERTM(shiftLUT.numel() == nNeurons, "shift and number of neurons must be same");
		
		shift<float>(output.data<float>(), input.data<float>(), shiftLUT.data<float>(), signalSize, nNeurons, nBatch, Ts);
	}

	return output;
}

torch::Tensor shift1Cuda(torch::Tensor input, torch::Tensor shiftLUT)
{
	return shiftCuda(input, shiftLUT, 1.0f);
}

torch::Tensor shiftFlCuda(torch::Tensor input, float shiftLUT, float Ts)
{
	CHECK_INPUT(input);

	cudaSetDevice(input.device().index());

	auto output = torch::empty_like(input);

	unsigned signalSize = input.size(-1);
	unsigned nNeurons   = input.numel()/signalSize;
	
	shift<float>(output.data<float>(), input.data<float>(), shiftLUT, signalSize, nNeurons, Ts);		
	
	return output;
}

torch::Tensor shiftFl1Cuda(torch::Tensor input, float shiftLUT)
{
	return shiftFlCuda(input, shiftLUT, 1.0f);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("getSpikes", &getSpikesCuda, "Get spikes (CUDA)");
	m.def("conv"     , &convCuda     , "Convolution in time (CUDA)");
	m.def("corr"     , &corrCuda     , "Correlation in time (CUDA)");
	m.def("shift"    , &shiftCuda    , "Element shift in time (CUDA)");
	m.def("shift"    , &shift1Cuda   , "Element shift in time (CUDA)");
	m.def("shift"    , &shiftFlCuda  , "Element shift in time (CUDA)");
	m.def("shift"    , &shiftFl1Cuda , "Element shift in time (CUDA)");
}
