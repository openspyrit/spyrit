# ==================================================================================
# Data consistency
# ==================================================================================
# ==================================================================================
class Pinv_orthogonal(nn.Module): # For A.T @ A  = n*Id (ex : Hadamard, Fourier...
# ==================================================================================
    def __init__(self):
        super().__init__()
        # FO = Forward Operator
        #-- Pseudo-inverse to determine levels of noise.
        
    def forward(self, x, FO):
        # input (b*c, M)
        # output (b*c, N)
        x = (1/FO.N)*FO.adjoint(x);
        return x


# ==================================================================================
class learned_measurement_to_image(nn.Module):
# ==================================================================================
    def __init__(self, N, M):
        super().__init__()
        # FO = Forward Operator
        self.FC = nn.Linear(M, N, True) # FC - fully connected
        
    def forward(self, x, FO = None):
        # input (b*c, M)
        # output (b*c, N)
        x = self.FC(x);
        return x
 
# ==================================================================================
class gradient_step(nn.Module):
# ==================================================================================
    def __init__(self, mu = 0.1):
        super().__init__()
        # FO = Forward Operator
        #-- Pseudo-inverse to determine levels of noise.
        self.mu = nn.Parameter(torch.tensor([mu], requires_grad=True)) #need device maybe?
        # if user wishes to keep mu constant, then he can change requires gard to false 
        
    def forward(self, x, x_0, FO):
        # x - input (b*c, M) - measurement vector
        # x_0 - input (b*c, N) - previous estimate
        # z - output (b*c, N)
        # z = x_0 - mu*A^T(A*x_0-x)
        x = FO.Forward_op(x_0)-x;
        x = x_0 - self.mu*FO.adjoint(x);
        return x
 
# ==================================================================================
class Tikhonov_cg(nn.Module):
# ==================================================================================
    def __init__(self, n_iter = 5, mu = 0.1, eps = 1e-6):
        super().__init__()
        # FO = Forward Operator - Works for ANY forward operator
        self.n_iter = n_iter;
        self.mu = nn.Parameter(torch.tensor([float(mu)], requires_grad=True)) #need device maybe?
        # if user wishes to keep mu constant, then he can change requires gard to false 
        self.eps = eps;
        # self.FO = FO

    def A(self,x, FO):
        return FO.Forward_op(FO.adjoint(x)) + self.mu*x

    def CG(self, y, FO, shape, device):
        x = torch.zeros(shape).to(device); 
        r = y - self.A(x, FO);
        c = r.clone()
        kold = torch.sum(r * r)
        a = torch.ones((1));
        for i in range(self.n_iter): 
            if a>self.eps : # Necessary to avoid numerical issues with a = 0 -> a = NaN
                Ac = self.A(c, FO)
                cAc =  torch.sum(c * Ac)
                a =  kold / cAc
                x += a * c
                r -= a * Ac
                k = torch.sum(r * r)
                b = k / kold
                c = r + b * c
                kold = k
        return x
        
    def forward(self, x, x_0, FO):
        # x - input (b*c, M) - measurement vector
        # x_0 - input (b*c, N) - previous estimate
        # z - output (b*c, N)
        # n_step steps of Conjugate gradient to solve \|Ax-b\|^2 + mu \|x - x_0\|^2
        # FO could be inside the class 

        y = x-FO.Forward_op(x_0);
        x = self.CG(y, FO, x.shape, x.device);
        x = x_0 + FO.adjoint(x)
        return x
#        
#    def forward(self, x, x_0):
#        # x - input (b*c, M) - measurement vector
#        # x_0 - input (b*c, N) - previous estimate
#        # z - output (b*c, N)
#        # n_step steps of Conjugate gradient to solve \|Ax-b\|^2 + mu \|x - x_0\|^2
#
#        y = x-self.FO.Forward_op(x_0);
#        x = self.CG(y, x.shape, x.device);
#        x = x_0 + self.FO.adjoint(x)
#        return x
#

# ==================================================================================
class Tikhonov_solve(nn.Module):
# ==================================================================================
    def __init__(self, mu = 0.1):
        super().__init__()
        # FO = Forward Operator - Needs to be matrix-storing
        #-- Pseudo-inverse to determine levels of noise.
        self.mu = nn.Parameter(torch.tensor([float(mu)], requires_grad=True)) #need device maybe?
    
    def solve(self, x, FO):
        A = FO.Mat()@torch.transpose(FO.Mat(), 0,1)+self.mu*torch.eye(FO.M); # Can precompute H@H.T to save time!
        A = A.view(1, FO.M, FO.M); # Instead of reshaping A, reshape x in the batch-final dimension
        #A = A.repeat(x.shape[0],1, 1); # Not optimal in terms of memory
        A = A.expand(x.shape[0],-1, -1); # Not optimal in terms of memory
        x = torch.linalg.solve(A, x);
        return x;

    def forward(self, x, x_0, FO):
        # x - input (b*c, M) - measurement vector
        # x_0 - input (b*c, N) - previous estimate
        # z - output (b*c, N)
        
        # uses torch.linalg.solve [As of Pytorch 1.9 autograd supports solve!!]
        x = x - FO.Forward_op(x_0);
        x = self.solve(x, FO);
        x = x_0 + FO.adjoint(x)
        return x

# ==================================================================================
class Orthogonal_Tikhonov(nn.Module):
# ==================================================================================
    def __init__(self, mu = 0.1):
        super().__init__()
        # FO = Forward Operator
        #-- Pseudo-inverse to determine levels of noise.
        self.mu = nn.Parameter(torch.tensor([float(mu)], requires_grad=True)) #need device maybe?
        
    def forward(self, x, x_0, FO):
        # x - input (b*c, M) - measurement vector
        # x_0 - input (b*c, N) - previous estimate
        # z - output (b*c, N)
        
        x = x - FO.Forward_op(x_0);
        x = x*(1/(FO.N+self.mu));# for hadamard, otherwise, line above
        x = FO.adjoint(x) + x_0;
        return x;


# ==================================================================================
class Generalised_Tikhonov_cg(nn.Module):# not inheriting from Tikhonov_cg because 
#                           the of the way var is called in CG
# ==================================================================================
    def __init__(self, Sigma_prior, n_iter = 6, eps = 1e-6):
        super().__init__()
        # FO = Forward Operator - Works for ANY forward operator
        # if user wishes to keep mu constant, then he can change requires gard to false 
        self.n_iter = n_iter;

        self.Sigma_prior = nn.Linear(Sigma_prior.shape[1], Sigma_prior.shape[0], False); 
        self.Sigma_prior.weight.data=torch.from_numpy(Sigma_prior)
        self.Sigma_prior.weight.data=self.Sigma_prior.weight.data.float()
        self.Sigma_prior.weight.requires_grad=False
        self.eps = eps;


    def A(self,x, var, FO):
        return FO.Forward_op(self.Sigma_prior(FO.adjoint(x))) + torch.mul(x,var); # the first part can be precomputed for optimisation

    def CG(self, y, var, FO, shape, device):
        x = torch.zeros(shape).to(device); 
        r = y - self.A(x, var, FO);
        c = r.clone()
        kold = torch.sum(r * r)
        a = torch.ones((1));
        for i in range(self.n_iter):
            if a > self.eps :
                Ac = self.A(c, var, FO)
                cAc =  torch.sum(c * Ac)
                a =  kold / cAc
                x += a * c
                r -= a * Ac
                k = torch.sum(r * r)
                b = k / kold
                c = r + b * c
                kold = k
        return x
        
    def forward(self, x, x_0, var_noise, FO):
        # x - input (b*c, M) - measurement vector
        # x_0 - input (b*c, N) - previous estimate
        # var_noise - input (b*c, M) - estimated variance of noise
        # z - output (b*c, N)
        # n_step steps of Conjugate gradient to solve 
        # \|Ax-b\|^2_{sigma_prior^-1} + \|x - x_0\|^2_{var_noise^-1}
        y = x-FO.Forward_op(x_0);
        x = self.CG(y, var_noise, FO, x.shape, x.device);
        x = x_0 + self.Sigma_prior(FO.adjoint(x))
        return x


# ==================================================================================
class Generalised_Tikhonov_solve(nn.Module):
# ==================================================================================
    def __init__(self, Sigma_prior):
        super().__init__()
        # FO = Forward Operator - Needs to be matrix-storing
        # -- Pseudo-inverse to determine levels of noise.
        self.Sigma_prior = nn.Parameter(\
                torch.from_numpy(Sigma_prior.astype("float32")), requires_grad=True)

    def solve(self, x, var, FO):
        A = FO.Mat() @ self.Sigma_prior @ torch.transpose(FO.Mat(), 0,1)
        A = A.view(1, FO.M, FO.M);
        #A = A.repeat(x.shape[0],1,1);# this could be precomputed maybe
        #A += torch.diag_embed(var);
        A = A.expand(x.shape[0],-1,-1) + torch.diag_embed(var);
        x = torch.linalg.solve(A, x);
        return x;

    def forward(self, x, x_0, var_noise, FO):
        # x - input (b*c, M) - measurement vector
        # x_0 - input (b*c, N) - previous estimate
        # z - output (b*c, N)
        
        # uses torch.linalg.solve [As of Pytorch 1.9 autograd supports solve!!]
        # torch linal solve uses (I believe the LU decomposition of matrix A to
        # solve the linear system.

        x = x - FO.Forward_op(x_0);
        x = self.solve(x, var_noise, FO);
        x = x_0 + torch.matmul(self.Sigma_prior,FO.adjoint(x).T).T
        return x


# ===========================================================================================
class Generalized_Orthogonal_Tikhonov(nn.Module): # todo: rename with _diag
#class Tikhonov_Orthogonal_Diag(nn.Module):
# ===========================================================================================   
    def __init__(self, sigma_prior, M, N):
        super().__init__()
        # FO = Forward Operator - needs foward operator with full inverse transform
        #-- Pseudo-inverse to determine levels of noise.
        
        self.comp = nn.Linear(M, N-M, False)
        self.denoise_layer = Denoise_layer(M);
        
        diag_index = np.diag_indices(N);
        var_prior = sigma_prior[diag_index];
        var_prior = var_prior[:M]

        self.denoise_layer.weight.data = torch.from_numpy(np.sqrt(var_prior));
        self.denoise_layer.weight.data = self.denoise_layer.weight.data.float();
        self.denoise_layer.weight.requires_grad = False

        Sigma1 = sigma_prior[:M,:M];
        Sigma21 = sigma_prior[M:,:M];
        W = Sigma21 @ np.linalg.inv(Sigma1);
        
        self.comp.weight.data=torch.from_numpy(W)
        self.comp.weight.data=self.comp.weight.data.float()
        self.comp.weight.requires_grad=False
        
    def forward(self, x, x_0, var, FO):
        # x - input (b*c, M) - measurement vector
        # var - input (b*c, M) - measurement variance
        # x_0 - input (b*c, N) - previous estimate
        # output has dimension (b*c, N)
        #

        x = x - FO.Forward_op(x_0)
        y1 = torch.mul(self.denoise_layer(var),x) # Should be in denoising layer
        y2 = self.comp(y1)

        y = torch.cat((y1,y2),-1)
        x = x_0 + FO.inverse(y) 
        return x

# ===========================================================================================
class List_Generalized_Orthogonal_Tikhonov(nn.Module): 
# ===========================================================================================   
    def __init__(self, sigma_prior_list, M, N, n_comp = None, n_denoi=None):
        super().__init__()
        # FO = Forward Operator - needs foward operator with defined inverse transform
        #-- Pseudo-inverse to determine levels of noise.
       
        if n_denoi is None :
            n_denoi = len(sigma_prior_list)
        self.n_denoi = n_denoi
       
        if n_comp is None :
            n_comp = len(sigma_prior_list)
        self.n_comp = n_comp
      

        comp_list = [];
        for i in range(self.n_comp):
            comp_list.append(nn.Linear(M, N-M, False))
            
            index = min(i,len(sigma_prior_list)-1)
            Sigma1 = sigma_prior_list[index][:M,:M];
            Sigma21 = sigma_prior_list[index][M:,:M];
            
            W = Sigma21@np.linalg.inv(Sigma1);

            comp_list[i].weight.data=torch.from_numpy(W)
            comp_list[i].weight.data=comp_list[i].weight.data.float()
            comp_list[i].weight.requires_grad=False
 
        self.comp_list = nn.ModuleList(comp_list);
       
        denoise_list = [];
        for i in range(self.n_denoi):
            denoise_list.append(Denoise_layer(M))
            
            index = min(i,len(sigma_prior_list)-1)
        
            diag_index = np.diag_indices(N);
            var_prior = sigma_prior_list[index][diag_index];
            var_prior = var_prior[:M]
     
            denoise_list[i].weight.data = torch.from_numpy(np.sqrt(var_prior));
            denoise_list[i].weight.data = denoise_list[i].weight.data.float();
            denoise_list[i].weight.requires_grad = True;
        self.denoise_list = nn.ModuleList(denoise_list);
 
     
    def forward(self, x, x_0, var, FO, iterate):
        # x - input (b*c, M) - measurement vector
        # var - input (b*c, M) - measurement variance
        # x_0 - input (b*c, N) - previous estimate
        # z - output (b*c, N)
        #

        i = min(iterate, self.n_denoi-1)
        j = min(iterate, self.n_comp-1)

        x = x - FO.Forward_op(x_0);
        y1 = torch.mul(self.denoise_list[i](var),x);
        y2 = self.comp_list[j](y1);

        y = torch.cat((y1,y2),-1);
        x = x_0+FO.inverse(y) 
        return x;


# ===========================================================================================
class Denoise_layer(nn.Module):
# ===========================================================================================
    r"""Applies a transformation to the incoming data: :math:`y = A^2/(A^2+x) `

    Args:
        in_features: size of each input sample

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{in})`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, 1)`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`

    Examples::
        >>> m = Denoise_layer(30)
        >>> input = torch.randn(128, 30)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    def __init__(self, in_features):
        super(Denoise_layer, self).__init__()
        self.in_features = in_features
        self.weight = nn.Parameter(torch.Tensor(in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight, 0, 2/math.sqrt(self.in_features))

    def forward(self, inputs):
        return tikho(inputs, self.weight)

    def extra_repr(self):
        return 'in_features={}'.format(self.in_features)

def tikho(inputs, weight):
    # type: (Tensor, Tensor) -> Tensor
    r"""
    Applies a transformation to the incoming data: :math:`y = A^2/(A^2+x)`.

    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions - Variance of measurements
        - Weight: :math:`(in\_features)` - corresponds to the standard deviation
          of our prior.
        - Output: :math:`(N, in\_features)`
    """
    sigma = weight**2; # prefer to square it, because when leant, it can got to the 
    #negative, which we do not want to happen.
    # TO BE Potentially done : square inputs.
    den = sigma + inputs;
    ret = sigma/den;
    return ret
