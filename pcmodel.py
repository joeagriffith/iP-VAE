import torch
import torch.nn as nn

# MNIST ONLY
class iPVAE(nn.Module):
    def __init__(self, z_features, alpha=1.0, mode='mlp'):
        super(iPVAE, self).__init__()
        # self.alpha = nn.Parameter(torch.tensor([alpha]))
        self.alpha = alpha
        self.z_features = z_features
        
        self.mode = mode
        if mode == 'mlp':
            # self.u0 = nn.Parameter(torch.rand(1, z_features))
            self.decoder = nn.Sequential(
                nn.Linear(z_features, 512, bias=True),
                nn.ReLU(),
                nn.Linear(512, 512, bias=True),
                nn.ReLU(),
                nn.Linear(512, 784, bias=True),
                nn.Sigmoid(),
            )
        elif mode == 'cnn':
            self.initial_state = nn.Parameter(torch.randn(1, z_features, 1, 1)*0.02)
            x_features = 1
            self.decoder = nn.Sequential(
                nn.Unflatten(1, (z_features, 1, 1)),
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
            raise(f'mode must be "mlp" or "cnn", was "{mode}"')


    def forward(self, x, steps=16, losses=False):
        N = x.shape[0]
        if self.mode == 'mlp' and x.dim() > 2:
            x = x.flatten(1)

        L_recon = []

        u = torch.randn(N, self.z_features, device=x.device)
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
            for _ in range(steps):
                u = u.detach().requires_grad_(True)
                x_pred = self.decoder(u)
                mse = (x - x_pred).pow(2).flatten(1).sum(-1).mean()
                du = -0.5*torch.autograd.grad(mse, u, create_graph=True)[0]
                u = u + self.alpha * du

                x_pred = self.decoder(u)
                mse = (x - x_pred).pow(2).flatten(1).sum(-1).mean()

                if losses:
                    L_recon.append(mse)

        if losses:
            return u, L_recon, [torch.tensor([0], device=x.device)]
        else:
            return u