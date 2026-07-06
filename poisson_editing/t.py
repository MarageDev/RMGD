from torchvision.utils import save_image
from torchvision.transforms import transforms
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
im = Image.open('poisson_editing/im2.png')
transform = transforms.Compose([transforms.ToTensor()])
t_im = transform(im)
def tensor_to_numpy_img(tensor):
    return np.clip(tensor.permute(1, 2, 0).cpu().detach().numpy(), 0, 1)
w = torch.ones_like(t_im)
b = torch.zeros_like(t_im)

m_im = torch.where(torch.norm(t_im,p=2.,dim=0) > 1.9, b, w)
plt.imshow(tensor_to_numpy_img(m_im[:3,:,:]))
plt.imsave("poisson_editing/im_omega.png", tensor_to_numpy_img(m_im[:3,:,:]))
plt.show()