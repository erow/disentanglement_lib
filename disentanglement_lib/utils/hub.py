import os

import torch
import wandb
import gin


def get_model(run,
              device='cpu',
              conf="train.gin",
              model_file="model.pt"):
    from disentanglement_lib.methods.unsupervised.train import Train
    if isinstance(run, str):
        model_path = os.path.join(run, model_file)
        conf_path = os.path.join(run, conf)
    else:
        model_path = run.file(model_file).download('tmp', True).name
        conf_path = run.file(conf).download('tmp', True).name
    with gin.unlock_config():
        gin.parse_config_file(conf_path, True)
    train = Train()
    model = train.ae
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model


def convert_model(model, device='cpu'):
    def _decoder(latent_vectors):
        with torch.no_grad():
            z = torch.FloatTensor(latent_vectors).to(device)
            torch_imgs = model.decode(z).cpu().numpy()
            return torch_imgs.transpose((0, 2, 3, 1))

    def _encoder(obs):
        with torch.no_grad():
            # if isinstance(obs,torch.Tensor):
            obs = torch.FloatTensor(obs.transpose((0, 3, 1, 2))).to(device)  # convert tf format to torch's
            mu, logvar = model.encode(obs)
            mu, logvar = mu.cpu().numpy(), logvar.cpu().numpy()
            return mu, logvar

    return _encoder, _decoder
