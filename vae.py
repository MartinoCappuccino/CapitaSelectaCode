import torch
import torch.nn as nn

CONV = lambda in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, : \
    nn.Conv2d(in_channels, out_channels, kernel_size, stride  , padding  , bias=bias, padding_mode="reflect")
BATCH_NORM = nn.InstanceNorm2d
RELU = lambda : nn.LeakyReLU(negative_slope=0.2, inplace=True)

class ResidualBlock(nn.Module):
    """Basic convolutional building block
    Parameters
    ----------
    in_ch : int
        number of input channels to the block
    out_ch : int     
        number of output channels of the block
    """

    def __init__(self, in_ch, out_ch, stride=1, upsample=1):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=upsample)
        self.conv1 = CONV(in_ch , out_ch, 3, stride, 1, bias=False)
        self.relu =  RELU()
        self.bn1 = BATCH_NORM(out_ch)
        self.conv2 = CONV(out_ch, out_ch, 3,      1, 1, bias=False)
        self.bn2 = BATCH_NORM(out_ch)
        self.shortcut = []
        if stride!=1 or in_ch!=out_ch:
            self.shortcut.append(CONV(in_ch, out_ch, 1, stride, bias=True))
            self.shortcut.append(BATCH_NORM(out_ch))
        self.shortcut = nn.Sequential(*self.shortcut)

    def forward(self, x):
        """Performs a forward pass of the block
       
        x : torch.Tensor
            the input to the block
        torch.Tensor
            the output of the forward pass
        """
        residual = self.shortcut(x)
        residual = self.upsample(residual)
        out = self.upsample(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.relu(out)
        return out

rec_loss = nn.L1Loss()


class Encoder(nn.Module):
    """The encoder part of the VAE.
    Parameters
    ----------
    spatial_size : list[int]
        size of the input image, by default [64, 64]
    z_dim : int
        dimension of the latent space
    chs : tuple
        hold the number of input channels for each encoder block
    """

    def __init__(self, spatial_size=[64, 64], z_dim=256, chs=(1, 32, 64, 128, 256)):
        super().__init__()
        self.conv1 = CONV(chs[0], chs[1], 7, 1, 3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.enc_blocks = nn.ModuleList(
            [ResidualBlock(chs[i], chs[i+1], 2, 1) for i in range(1,len(chs)-1)]
        )
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # height and width of images at lowest resolution level
        _h, _w = spatial_size[0]//(2**len(chs)), spatial_size[1]//(2**len(chs))

        # flattening
        self.final = nn.Sequential(nn.Flatten(1), nn.Linear(chs[-1] * _h * _w, 2 * z_dim))

    def forward(self, x):
        """Performs the forward pass for all blocks in the encoder.
        Parameters
        ----------
        x : torch.Tensor
            input image to the encoder
        Returns
        -------
        list[torch.Tensor]    
            a tensor with the means and a tensor with the log variances of the
            latent distribution
        """
        out = self.conv1(x)
        out = self.maxpool(out)
        for block in self.enc_blocks:
            out = block(out)
        out = self.avgpool(out)
        out = self.final(out)
        return torch.chunk(out, 2, dim=1)  # 2 chunks, 1 each for mu and logvar


class Generator(nn.Module):
    """Generator of the GAN
    Parameters
    ----------
    z_dim : int 
        dimension of latent space
    chs : tuple
        holds the number of channels for each block
    h : int, optional
        height of image at lowest resolution level, by default 2
    w : int, optional
        width of image at lowest resolution level, by default 2
    """

    def __init__(self, z_dim=256, chs=(256, 128, 64, 32, 1), h=2, w=2):

        super().__init__()
        self.chs = chs
        self.h = h  
        self.w = w  
        self.z_dim = z_dim  
        self.proj_z = nn.Linear(
            self.z_dim, self.chs[0] * self.h * self.w
        )  # fully connected layer on latent space
        self.reshape = lambda x: torch.reshape(
            x, (-1, self.chs[0], self.h, self.w)
        )  # reshaping
        self.dec_blocks = nn.ModuleList(
            [ResidualBlock(chs[0], chs[0]  , 1, 2)] + \
            [ResidualBlock(chs[i], chs[i+1], 1, 2) for i in range(0,len(chs)-2)]
        )
        self.upsample = nn.Upsample(scale_factor=2)
        self.proj_o = nn.Sequential(
            CONV(chs[-2], chs[-1], 3, 1, 1),
            nn.Tanh(),
        )  # output layer with Tanh activation

    def forward(self, z):
        """Performs the forward pass of decoder
        Parameters
        ----------
        z : torch.Tensor
            input to the generator
        
        Returns
        -------
        x : torch.Tensor
        
        """
        out = self.proj_z(z)
        out = self.reshape(out)
        for block in self.dec_blocks:
            out = block(out)
        out = self.upsample(out)
        out = self.proj_o(out)
        return out


class VAE(nn.Module):
    """A representation of the VAE
    Parameters
    ----------
    enc_chs : tuple 
        holds the number of input channels of each block in the encoder
    dec_chs : tuple 
        holds the number of input channels of each block in the decoder
    """
    def __init__(
        self,
        enc_chs=(1, 32, 64, 128, 256),
        dec_chs=(256, 128, 64, 32, 1),
    ):
        super().__init__()
        self.encoder = Encoder()
        self.generator = Generator()

    def forward(self, x):
        """Performs a forwards pass of the VAE and returns the reconstruction
        and mean + logvar.
        Parameters
        ----------
        x : torch.Tensor
            the input to the encoder
        Returns
        -------
        torch.Tensor
            the reconstruction of the input image
        float
            the mean of the latent distribution
        float
            the log of the variance of the latent distribution
        """
        mu, logvar = self.encoder(x)
        latent_z = sample_z(mu, logvar)

        return self.generator(latent_z), mu, logvar


def get_noise(n_samples, z_dim, device="cpu"):
    """Creates noise vectors.
    
    Given the dimensions (n_samples, z_dim), creates a tensor of that shape filled with 
    random numbers from the normal distribution.
    Parameters
    ----------
    n_samples : int
        the number of samples to generate
    z_dim : int
        the dimension of the noise vector
    device : str
        the type of the device, by default "cpu"
    """
    return torch.randn(n_samples, z_dim, device=device)


def sample_z(mu, logvar):
    """Samples noise vector from a Gaussian distribution with reparameterization trick.
    Parameters
    ----------
    mu : float
        the mean of the distribution
    logvar : float
        the log of the variance of the distribution
    """
    eps = torch.randn(mu.size(), device=mu.device).to(mu.dtype)
    return (logvar / 2).exp() * eps + mu


def kld_loss(mu, logvar):
    """Computes the KLD loss given parameters of the predicted 
    latent distribution.
    Parameters
    ----------
    mu : float
        the mean of the distribution
    logvar : float
        the log of the variance of the distribution
    Returns
    -------
    float
        the kld loss
    """
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def vae_loss(inputs, recons, mu, logvar):
    """Computes the VAE loss, sum of reconstruction and KLD loss
    Parameters
    ----------
    inputs : torch.Tensor
        the input images to the vae
    recons : torch.Tensor
        the predicted reconstructions from the vae
    mu : float
        the predicted mean of the latent distribution
    logvar : float
        the predicted log of the variance of the latent distribution
    Returns
    -------
    float
        sum of reconstruction and KLD loss
    """
    return rec_loss(inputs, recons) + kld_loss(mu, logvar)
