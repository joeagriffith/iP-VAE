import torch
import torch.nn as nn

# MNIST ONLY
class iPVAE(nn.Module):
    def __init__(self, z_features, alpha=0.1, mode='mlp'):
        super(iPVAE, self).__init__()
        self.alpha = alpha
        
        self.mode = mode
        if mode == 'mlp':
            self.initial_state = nn.Parameter(torch.randn(1, z_features)*0.02)
            self.decoder = nn.Sequential(
                nn.Linear(z_features, 784)
            )
            # self.decoder = nn.Sequential(
            #     nn.Linear(z_features, 1024),
            #     nn.ReLU(),
            #     nn.Linear(1024, 784),
            #     nn.Sigmoid(),
            # )
        elif mode == 'conv':
            self.initial_state = nn.Parameter(torch.randn(1, z_features, 1, 1)*0.02)
            x_features = 1
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(z_features, max(256, x_features), 3, 1),
                nn.ReLU(),

                nn.ConvTranspose2d(max(256, x_features), max(128, x_features), 3, 3),
                nn.ReLU(),
                
                nn.ConvTranspose2d(max(128, x_features), max(64, x_features), 3, 3),
                nn.ReLU(),
                
                nn.ConvTranspose2d(max(64, x_features), max(32, x_features), 2, 1),

                nn.ReLU(),
                nn.Conv2d(max(32, x_features), x_features, 3, 1, 1),
            )
        else:
            raise(f'mode must be "mlp" or "conv", was "{mode}"')


    def forward(self, x, steps=16, collect_errs=False):
        batch_size = x.shape[0]
        if self.mode == 'mlp' and x.dim() > 2:
            x = x.flatten(1)
        u = self.initial_state
        u = u.repeat(batch_size, *([1] * (u.dim() - 1)))

        errs = []
        for _ in range(steps):
            r = torch.exp(u)
            z = torch.poisson(r)
            z.requires_grad_(True)
            pred = self.decoder(z)
            # e = (x - pred).pow(2).mean()

            # du = torch.autograd.grad(e, z)[0]
            # u = u - self.alpha*du

            # e.backward()
            # u = u - self.alpha*z.grad

            e = x - pred
            du = e @ self.decoder[0].weight
            u = u - self.alpha*du

            if collect_errs:
                errs.append(e.item())

        if collect_errs:
            return z, errs
        else:
            return z
    
    def loss(self, x, steps=16):
        if self.mode == 'mlp' and x.dim() > 2:
            x = x.flatten(1)
        z = self.forward(x, steps=steps)
        pred = self.decoder(z)
        return (x - pred).pow(2).mean()
