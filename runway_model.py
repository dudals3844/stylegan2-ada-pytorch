import runway
import pickle
import numpy as np
import torch
from runway import image
from runway.data_types import category, file


@runway.setup(options={'checkpoint': file(extension=".pkl")})
def setup(opts):
    global Gs
    print(f"opt: {opts}")
    # if opts['model_selection'] == 'Big':
    #     opts['checkpoint'] = './pretrain/big_flower.pkl'
    # elif opts['model_selection'] == 'Small':
    opts['checkpoint'] = './pretrain/small_flower.pkl'
    # else:
    #     raise Exception('No Module Path')
    with open(opts['checkpoint'], 'rb') as file:
        Gs = pickle.load(file)
    return Gs


generate_inputs = {
    'z': runway.vector(512, sampling_std=0.5),
    'truncation': runway.number(min=0, max=1, default=0.8, step=0.01),
    "model_selection": category(choices=['Big', 'Small', 'Default'], default="Default")
}


@runway.command('generate', inputs=generate_inputs,
                outputs={'image': runway.image})
def convert(model, inputs):
    z = inputs['z']
    truncation = inputs['truncation']
    latents = z.reshape((1, 512))

    device = torch.device('cuda')

    if inputs['model_selection'] == 'Big':
        inputs['checkpoint'] = './pretrain/big_flower.pkl'
        with open(inputs['checkpoint'], 'rb') as file:
            model = pickle.load(file)
    elif inputs['model_selection'] == 'Small':
        inputs['checkpoint'] = './pretrain/small_flower.pkl'
        with open(inputs['checkpoint'], 'rb') as file:
            model = pickle.load(file)
    elif inputs['model_selection'] == 'Default':
        model = model
    else:
        raise Exception('No Module Path')

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
