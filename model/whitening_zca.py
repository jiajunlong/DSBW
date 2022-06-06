import torch
import torch.nn as nn
from torch.nn.functional import conv2d
class _Whitening(nn.Module):

	def __init__(self, num_features, w, h, running_m=None, running_var=None, momentum=0.1, track_running_stats=True, eps=1e-3):
		super(_Whitening, self).__init__()
		self.num_features = num_features
		self.momentum = momentum
		self.track_running_stats = track_running_stats
		self.eps = eps
		self.w = w
		self.h = h
		self.group_size = 16
		self.num_groups = 32
		self.running_m = running_m
		self.running_var = running_var

		if self.track_running_stats and self.running_m is not None:
			self.register_buffer('running_mean', self.running_m)
			self.register_buffer('running_variance', self.running_var)
		else:
			self.register_buffer('running_mean', torch.zeros([1, self.num_groups, 1, 1], out=torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()))
			self.register_buffer('running_variance', torch.ones([self.num_groups, w*h, w*h], out=torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()))
		
	def _check_input_dim(self, input):
		raise NotImplementedError


	def forward(self, x):

		# 将每个channel内将w,h作为特征，batchsize作为样本个数进行白化   

		self._check_input_dim(x)
        #x = (b,c,w,h)  c w*h => (1,c,1,1)
		b,c,w,h = xn.shape
		x = x.view(b*self.group_size,self.num_groups,w,h)
		m = x.mean(0).view(self.num_groups, -1).mean(-1).view(1, -1, 1, 1)
		if not self.training and self.track_running_stats: # for inference
			m = self.running_mean
		xn = x - m
        # T = (b,c,w,h) => (n,s,-1)
		
		T = xn.permute(1,2,3,0).contiguous().view(self.num_groups, -1, self.group_size*b)
        # f_cov = (c, w*h, w*h)
		f_cov = torch.bmm(T, T.permute(0,2,1))
		f_cov_shrinked = (1-self.eps) * f_cov + self.eps * torch.eye(w*h, out=torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()).repeat(self.num_groups, 1, 1)

		if not self.training and self.track_running_stats: # for inference
			f_cov_shrinked = (1-self.eps) * self.running_variance + self.eps * torch.eye(w*h, out=torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()).repeat(self.num_groups, 1, 1)
		u, s, v = torch.svd(f_cov_shrinked)
		sqrt = torch.sqrt(s.view(self.num_groups, w*h, 1) * torch.ones((self.num_groups, w*h, b*self.group_size),out=torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()))
		pca = torch.bmm(u.permute(0, 2, 1), T) / sqrt
		zca = torch.bmm(u, pca)
		decorrelated = zca.permute(2,0,1).contiguous().view(b,c,w,h)

		if self.training and self.track_running_stats:
			self.running_mean = torch.add(self.momentum * m.detach(), (1 - self.momentum) * self.running_mean, out=self.running_mean) 
			self.running_variance = torch.add(self.momentum * f_cov.detach(), (1 - self.momentum) * self.running_variance, out=self.running_variance)
			
		return decorrelated

class WTransform2d(_Whitening):
	def _check_input_dim(self, input):
		if input.dim() != 4:
			raise ValueError('expected 4D input (got {}D input)'. format(input.dim()))


# class whitening_scale_shift(nn.Module):
# 	def __init__(self, planes, w, h, running_mean=None, running_variance=None, track_running_stats=True, affine=True):
# 		super(whitening_scale_shift, self).__init__()
# 		self.planes = planes
# 		self.w = w
# 		self.h = h
# 		self.track_running_stats = track_running_stats
# 		self.affine = affine
# 		self.running_mean = running_mean
# 		self.running_variance = running_variance

# 		self.wh = WTransform2d(self.planes, 
# 										 self.w, 
# 										 self.h,
# 										 running_m=self.running_mean, 
# 										 running_var=self.running_variance, 
# 										 track_running_stats=self.track_running_stats)
# 		if self.affine:
# 			self.gamma = nn.Parameter(torch.ones(self.planes, 1, 1))
# 			self.beta = nn.Parameter(torch.zeros(self.planes, 1, 1))

# 	def forward(self, x):
# 		out = self.wh(x)
# 		if self.affine:
# 			out = (out + self.beta) * self.gamma
# 		return out

class _Whitening(nn.Module):

	def __init__(self, num_features, group_size=4, running_m=None, running_var=None, momentum=0.1, track_running_stats=True, eps=1e-3):
		super(_Whitening, self).__init__()
		self.num_features = num_features
		self.momentum = momentum
		self.track_running_stats = track_running_stats
		self.eps = eps
		self.group_size = group_size
		self.num_groups = self.num_features // self.group_size
		self.running_m = running_m
		self.running_var = running_var

		if self.track_running_stats and self.running_m is not None:
			self.register_buffer('running_mean', self.running_m)
			self.register_buffer('running_variance', self.running_var)
		else:
			self.register_buffer('running_mean', torch.zeros([self.num_groups, self.group_size, 1, 1, 1], out=torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()))
			self.register_buffer('running_variance', torch.ones([self.num_groups, self.group_size, self.group_size], out=torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()))
		
	def _check_input_dim(self, input):
		raise NotImplementedError
        
	def _check_group_size(self):
		raise NotImplementedError

	def forward(self, x):

		# 将每个channel内将b,w,h作为样本进行白化   
		b,c,w,h = x.shape
		self._check_input_dim(x)
		if self.group_size != 1:
			self._check_group_size()
        #x = (b,c,w,h)   => (n,s,b,w,h)
		m = x.mean(0).view(self.num_features, -1).mean(-1).view(self.num_groups, self.group_size, 1, 1, 1)
		if not self.training and self.track_running_stats: # for inference
			m = self.running_mean
		xt = x.permute(1,0,2,3).contiguous().view(self.num_groups, self.group_size,b,w,h)        
		xn = xt - m
        # xn = (n,s,b,w,h)  T => (n,s,-1)
		T = xn.contiguous().view(self.num_groups, self.group_size,-1)
        # f_cov = (n, s, s)
		f_cov = torch.bmm(T, T.permute(0,2,1))
		f_cov_shrinked = (1-self.eps) * f_cov + self.eps * torch.eye(self.group_size, out=torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()).repeat(self.num_groups, 1, 1)

		if not self.training and self.track_running_stats: # for inference
			f_cov_shrinked = (1-self.eps) * self.running_variance + self.eps * torch.eye(self.group_size, out=torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()).repeat(self.num_groups, 1, 1)
		u, s, v = torch.svd(f_cov_shrinked)
		sqrt = (torch.sqrt(s.view(self.num_groups,self.group_size, 1)+self.eps*torch.ones((self.num_groups, self.group_size, 1),out=torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()))\
		 * torch.ones((self.num_groups, self.group_size, b*w*h),out=torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()))
		pca = torch.bmm(u.permute(0, 2, 1), T) / sqrt
		zca = torch.bmm(u, pca)
		decorrelated = zca.view(self.num_groups,self.group_size,b,w,h).view(c,b,w,h).permute(1,0,2,3)

		if self.training and self.track_running_stats:
			self.running_mean = torch.add(self.momentum * m.detach(), (1 - self.momentum) * self.running_mean, out=self.running_mean) 
			self.running_variance = torch.add(self.momentum * f_cov.detach(), (1 - self.momentum) * self.running_variance, out=self.running_variance)
			
		return decorrelated

class WTransform2d(_Whitening):
	def _check_input_dim(self, input):
		if input.dim() != 4:
			raise ValueError('expected 4D input (got {}D input)'. format(input.dim()))
	def _check_group_size(self):
		if self.num_features % self.group_size != 0:
			raise ValueError('expected number of channels divisible by group_size (got {} group_size\
				for {} number of features'.format(self.group_size, self.num_features))


class whitening_scale_shift(nn.Module):
	def __init__(self, planes, group_size=1, running_mean=None, running_variance=None, track_running_stats=True, affine=True):
		super(whitening_scale_shift, self).__init__()
		self.planes = planes
		self.group_size = group_size
		self.track_running_stats = track_running_stats
		self.affine = affine
		self.running_mean = running_mean
		self.running_variance = running_variance

		self.wh = WTransform2d(self.planes, 
										 self.group_size,
										 running_m=self.running_mean, 
										 running_var=self.running_variance, 
										 track_running_stats=self.track_running_stats)
		if self.affine:
			self.gamma = nn.Parameter(torch.ones(self.planes, 1, 1))
			self.beta = nn.Parameter(torch.zeros(self.planes, 1, 1))

	def forward(self, x):
		out = self.wh(x)
		if self.affine:
			out = out * self.gamma  + self.beta
		return out
