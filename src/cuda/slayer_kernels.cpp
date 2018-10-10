#include <torch/torch.h>

#include <vector>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Forward declarations, cuda function
std::vector<at::Tensor> getSpikesCuda(
	at::Tensor d_u,
	at::Tensor d_s,
	const at::Tensor d_nu,
	at::Tensor theta,
	at::Tensor Ts);

// C++ - Python interface
std::vector<at::Tensor> get_spikes_cuda(
	at::Tensor d_u,
	at::Tensor d_s,
	const at::Tensor d_nu,
	at::Tensor theta,
	at::Tensor Ts)
	{
		CHECK_INPUT(d_u);
		CHECK_INPUT(d_s);
		CHECK_INPUT(d_nu);
		return getSpikesCuda(d_u, d_s, d_nu, theta, Ts);
	}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("get_spikes_cuda", &get_spikes_cuda, "Get_spikes (CUDA)");
}