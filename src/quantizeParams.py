import torch

class quantizeWeights(torch.autograd.Function):
	@staticmethod
	def forward(ctx, weights, step=1):
		# return weights
		# print('Weights qunatized with step', step)
		return torch.round(weights / step) * step

	@staticmethod
	def backward(ctx, gradOutput):
		return gradOutput, None