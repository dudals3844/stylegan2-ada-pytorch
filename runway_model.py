import copy

import runway
import pickle
import numpy as np
import torch
from runway import image
import legacy
import dnnlib




@runway.setup(options={'checkpoint': runway.file(extension='.pkl')})
def setup(opts):
    global Gs
    opts['checkpoint'] = './pretrain/network-snapshot-000096.pkl'
    with open(opts['checkpoint'], 'rb') as file:
        Gs = pickle.load(file)
    return Gs

generate_inputs = {
    'z': runway.vector(512, sampling_std=0.5),
    'truncation': runway.number(min=0, max=1, default=0.8, step=0.01)
}

@runway.command('generate', inputs=generate_inputs, outputs={'image': runway.image})
def convert(model, inputs):
    z = inputs['z']
    truncation = inputs['truncation']
    latents = z.reshape((1, 512))
    device = torch.device('cuda')

    G = model['G_ema'].to(device)
    label = torch.zeros([1, G.c_dim], device=device)
    seed = np.random.randint(1000)

    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    img = G(z, label, truncation_psi=truncation, noise_mode='random')
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    output = img[0].cpu().numpy()
    return {'image': output}


if __name__ == '__main__':
    runway.run(host='0.0.0.0', port=3000)