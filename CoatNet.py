import torch
import torch.nn as nn


class SEBlock(nn.Module):
	"credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"

	def __init__(self, c, r=16):
		super().__init__()
		self.squeeze = nn.AdaptiveAvgPool2d(1)
		self.excitation = nn.Sequential(
			nn.Linear(c, c // r, bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(c // r, c, bias=False),
			nn.Sigmoid()
		)

	def forward(self, x):
		bs, c, _, _ = x.shape
		y = self.squeeze(x).view(bs, c)
		y = self.excitation(y).view(bs, c, 1, 1)
		return x * y.expand_as(x)


class MBConvolution(nn.Module):

	def __init__(self, k, kp, order, image_size):
		super().__init__()

		# must downsample on the first block of each stage
		if order == 0:
			self.downsample = True

		else:
			self.downsample = False
			# if have already done one in this block, the input will already have kp channels
			k = kp

		# paper calls for normalization step at the start of every block
		self.prenorm = nn.BatchNorm2d(k)

		# define stride to downsample if needed
		if self.downsample:
			stride = 2

			# make conv and pooling combination in appendix of CoAtNet, allows residual action during striding
			self.strider = nn.Sequential(nn.MaxPool2d(3, 2, 1), nn.Conv2d(k, kp, 1, 1, 0, bias=False))

		else:
			stride = 1

		# find number of channels to expand using rate of 4 as in CoAtNet paper
		tk = k * 4

		# define actual block
		self.mbconv = nn.Sequential(

			#  downscale
			nn.Conv2d(k, tk, 1, stride, 0),
			nn.GELU(),
			nn.BatchNorm2d(tk),

			# depthwise 3x3 conv
			nn.Conv2d(tk, tk, 3, 1, 1, groups=tk),
			nn.GELU(),
			nn.BatchNorm2d(tk),
			SEBlock(tk),

			# linear
			nn.Conv2d(tk, kp, 1, 1, 0),
		)

	def forward(self, x):
		#pre-norm
		x = self.prenorm(x)

		# do thing
		if self.downsample:
			return self.strider(x) + self.mbconv(x)
		else:
			return x + self.mbconv(x)


class RelAttention(nn.Module):
	def __init__(self, k, kp, image_size, downsample):
		super().__init__()

		# define things
		self.k = k
		self.kp = kp

		if downsample:
			# assume image is a square, take in input of convolutional mapping, output 1D array for attention
			self.q, self.k, self.v = [
				nn.Sequential(nn.Linear(k * image_size * image_size, kp * image_size * image_size), nn.GELU()) for i in
				range(3)]
		else:
			# 32 size heads because that is the shape needed for MHA as prescribed in the paper
			self.q, self.k, self.v = [
				nn.Sequential(nn.Linear(kp * image_size * image_size, kp * image_size * image_size), nn.GELU()) for i in
				range(3)]
		self.heads = (kp * image_size * image_size) // 32
		self.TranslationalWeight = nn.Parameter(torch.rand(63), requires_grad=True)  # weight should be dim/2 - 1 size
		
		# because vector/matrix multiplication preserves order, a row of q*t is equal to a row of j, and the i coordinate is the column number; therefore, a matrix of diagonal weight values actually accomplishes the otherwise hard operation simply
		# this peice of code creates a diagonal index for quick indexing
		index_weight = torch.tensor([*range(63)])
		TW_size = 63
		i_s = 32
		self.index = torch.tensor(sum(
				[torch.diag(torch.full([i_s], index_weight[TW_size - 1 - n].reshape(1)[0]), n - i_s)[:i_s, :i_s] for n in
				range(i_s)]) + sum([torch.diag(torch.full([i_s], index_weight[i_s - 1 - n].reshape(1)[0]), n)[:i_s, :i_s] for n in
				range(i_s)]), dtype=torch.long)

	def forward(self, x):

		x = x.reshape(x.shape[0], x.numel() // x.shape[0])

		# define final output matrix
		tx = torch.tensor([])

		# convinence
		kp = self.kp

		# loop through heads, channels are in intervals of 32 as the paper supplies
		q, k, v = [self.q(x).reshape(x.shape[0], self.heads, 32, 1), self.k(x).reshape(x.shape[0], self.heads, 1, 32),
				   self.v(x).reshape(x.shape[0], self.heads, 32)]


		indexed_weight = self.TranslationalWeight[self.index]

		# make softmax weight, divide by sqrt(d of k) on q*k only because it is ambigous in paper and that seems right.
		a = nn.functional.softmax((torch.matmul(q, k) / torch.sqrt(torch.tensor(32))) + indexed_weight)

		# do final multipliction of values and softmax weight
		return torch.matmul(a, v.reshape(x.shape[0], self.heads, 32, 1))

class Transformer(nn.Module):
	def __init__(self, k, kp, order, image_size):
		super().__init__()

		# define things
		self.kp = kp
		self.image_size = image_size

		if order == 0:
			self.downsample = True
		else:
			self.downsample = False

		# define norm
		self.prenorm = nn.LayerNorm([kp, image_size, image_size])

		if self.downsample:
			# image size is calculated per-stage, first block in stage will need to fit image size to expected and gets last stage's image size
			self.pool1 = nn.MaxPool2d(3, 2, 1)
			self.pool2 = nn.MaxPool2d(3, 2, 1)
			self.proj = nn.Conv2d(k, kp, 1, 1, 0, bias=False)
			self.prenorm = nn.LayerNorm([k, image_size, image_size])

		# attention
		self.attention = RelAttention(k, kp, image_size, self.downsample)

		# FFN is actually not defined in the paper so here I just use one FC layer as the traditional transformer uses just that
		self.ffn = nn.Sequential(nn.Linear(kp * image_size * image_size, kp * image_size * image_size), nn.GELU())

	def forward(self, x):
		# do propogation as described in paper appendix

		if self.downsample:
			r = self.proj(self.pool1(x))
			x = r + self.attention(self.prenorm(self.pool2(x))).reshape(r.shape)
		else:
			x = x + self.attention(self.prenorm(x)).reshape(x.shape)
		x = x.reshape(x.shape[0], x.numel() // x.shape[0]) + self.ffn(x.reshape(x.shape[0], x.numel() // x.shape[0]))
		return x.reshape(x.shape[0], self.kp, self.image_size, self.image_size)


class CoAtNet(nn.Module):
	def __init__(self, in_channels, in_size, num_blocks, channels, num_classes=10,
				 block=[MBConvolution, MBConvolution, Transformer, Transformer]):

		# idea for block num variation from github.com/chinhsuanwu's code
		super().__init__()

		# create dict for storing stages
		self.s = nn.ModuleDict({})

		# manually define s0
		self.s["0"] = nn.Sequential(
			nn.BatchNorm2d(in_channels),
			nn.Conv2d(in_channels, channels[0], 1, 2, 0),
			nn.GELU(),
			nn.BatchNorm2d(channels[0]),
			nn.Conv2d(channels[0], channels[0], 1, 1, 0),
			nn.GELU(),
			nn.BatchNorm2d(channels[0])
		)

		# loop through amount of stages(4)
		for sn in range(1, 5):
			print(sn)
			# define sequential of blocks with amount i, as defined by num_blocks arg. Blocks are initialized with channel sizes as thier args and image sizes going to them

			self.s[str(sn)] = nn.Sequential(
				*[block[sn - 1](channels[sn - 1], channels[sn], i, in_size // (2 ** (sn + 1))) for i in range(num_blocks[sn])])

		# define pool to make 1 element per channel
		self.pool = nn.AvgPool2d(in_size // 32, 1)

		self.fc = nn.Linear(channels[-1], num_classes)

	def forward(self, x):

		#loop over x with stages
		for sn in range(5):
			x = self.s[str(sn)](x)

		# reduce dimensionality
		x = self.pool(x).reshape(x.shape[0], x.shape[1])
		return self.fc(x)
