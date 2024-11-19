import torch
import torch.nn as nn

# MNIST ONLY
class iPVAE(nn.Module):
    def __init__(self, num_features, alpha=1.0, mode='mlp'):
        super(iPVAE, self).__init__()
        self.alpha = alpha
        self.num_features = num_features
        
        self.mode = mode
        if mode == 'mlp':
            self.decoder = nn.Sequential(
                nn.Linear(num_features, 512, bias=True),
                nn.ReLU(),
                nn.Linear(512, 512, bias=True),
                nn.ReLU(),
                nn.Linear(512, 784, bias=True),
                nn.Sigmoid(),
            )
        elif mode == 'cnn':
            self.initial_state = nn.Parameter(torch.randn(1, num_features, 1, 1)*0.02)
            self.decoder = nn.Sequential(
                nn.Unflatten(1, (num_features, 1, 1)),
                nn.ConvTranspose2d(num_features, 256, 3, 1),
                nn.ReLU(),

                nn.ConvTranspose2d(256, 128, 3, 3),
                nn.ReLU(),
                
                nn.ConvTranspose2d(128, 64, 3, 3),
                nn.ReLU(),
                
                nn.ConvTranspose2d(64, 32, 2, 1),

                nn.ReLU(),
                nn.Conv2d(32, 1, 3, 1, 1),
                nn.Sigmoid(),
            )
        else:
            raise(f'mode must be "mlp" or "cnn", was "{mode}"')


    def forward(self, x, T=16, losses=False):
        N = x.shape[0]
        if self.mode == 'mlp' and x.dim() > 2:
            x = x.flatten(1)

        L_recon = []
        L_kl = []

        u = torch.randn(N, self.num_features, device=x.device)
        for t in range(T):
            u = u.detach().requires_grad_(True)
            r = torch.exp(u)
            z_prior = torch.poisson(r)
            preds_prior = self.decoder(z_prior)
            mse = (x - preds_prior).pow(2).sum()
            
            du = -0.5*torch.autograd.grad(mse, z_prior, create_graph=True)[0]
            dr = torch.exp(du)
            lam = r * dr
            z_post = torch.poisson(lam)
            preds_post = self.decoder(z_post)
            u = u + self.alpha * du

            if losses:
                L_recon.append((x - preds_post).pow(2).flatten(1).sum(-1).mean())
                L_kl.append((r *(1 - dr + dr * torch.log(dr))).flatten(1).sum(-1).mean())

        if losses:
            return z_post, L_recon, L_kl
        else:
            return z_post

    # def forward_tracked(self, x, T=16):
    #     N = x.shape[0]
    #     if self.mode == 'mlp' and x.dim() > 2:
    #         x = x.flatten(1)

    #     state = {
    #         'u': [torch.randn(N, self.num_features, device=x.device)],
    #         'r': [],
    #         'z_prior': [],
    #         'preds_prior': [],
    #         'delta': [],
    #         'J': [],
    #         'mse': [],
    #         'du': [],
    #         'dr': [],
    #         'y': [],
    #         'z_post': [],
    #         'preds_post': [],
    #         'L_recon': [],
    #         'L_kl': []
    #     }

    #     for t in range(T):
    #         state['u'][-1] = state['u'][-1].detach().requires_grad_(True)
    #         state['r'].append(torch.exp(state['u'][-1]))
    #         state['z_prior'].append(torch.poisson(state['r'][-1]))
    #         state['preds_prior'].append(self.decoder(state['z_prior'][-1]))
    #         state['mse'].append((x - state['preds_prior'][-1]).pow(2).sum())
    #         state['du'].append(-0.5*torch.autograd.grad(state['mse'][-1], state['z_prior'][-1], create_graph=True)[0])
    #         state['dr'].append(torch.exp(state['du'][-1]))
    #         state['y'].append(state['r'][-1] * state['dr'][-1])
    #         state['z_post'].append(torch.poisson(state['y'][-1]))
    #         state['preds_post'].append(self.decoder(state['z_post'][-1]))
    #         state['L_recon'].append((x.flatten(1) - state['preds_post'][-1].flatten(1)).pow(2).sum(-1).mean())
    #         state['L_kl'].append((state['r'][-1] *(1 - state['dr'][-1] + state['dr'][-1] * torch.log(state['dr'][-1]))).sum(-1).mean())
    #         state['u'].append(state['u'][-1] + self.alpha * state['du'][-1])
        
    #     return state