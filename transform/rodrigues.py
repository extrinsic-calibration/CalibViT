import torch 

class Sinc:
    @staticmethod
    def sinc1(t):
        """ sinc1: t -> sin(t)/t """
        e = 0.01
        r = torch.zeros_like(t)
        a = torch.abs(t)

        s = a < e
        c = (s == 0)
        t2 = t[s] ** 2
        r[s] = 1 - t2/6*(1 - t2/20*(1 - t2/42))  # Taylor series O(t^8)
        r[c] = torch.sin(t[c]) / t[c]

        return r


    @staticmethod
    def sinc1_dt(t):
        """ d/dt(sinc1) """
        e = 0.01
        r = torch.zeros_like(t)
        a = torch.abs(t)

        s = a < e
        c = (s == 0)
        t2 = t ** 2
        r[s] = -t[s]/3*(1 - t2[s]/10*(1 - t2[s]/28*(1 - t2[s]/54)))  # Taylor series O(t^8)
        r[c] = torch.cos(t[c])/t[c] - torch.sin(t[c])/t2[c]

        return r

    @staticmethod
    def sinc1_dt_rt(t):
        """ d/dt(sinc1) / t """
        e = 0.01
        r = torch.zeros_like(t)
        a = torch.abs(t)

        s = a < e
        c = (s == 0)
        t2 = t ** 2
        r[s] = -1/3*(1 - t2[s]/10*(1 - t2[s]/28*(1 - t2[s]/54)))  # Taylor series O(t^8)
        r[c] = (torch.cos(t[c]) / t[c] - torch.sin(t[c]) / t2[c]) / t[c]

        return r

    
    @staticmethod
    def rsinc1(t):
        """ rsinc1: t -> t/sinc1(t) """
        e = 0.01
        r = torch.zeros_like(t)
        a = torch.abs(t)

        s = a < e
        c = (s == 0)
        t2 = t[s] ** 2
        r[s] = (((31*t2)/42 + 7)*t2/60 + 1)*t2/6 + 1  # Taylor series O(t^8)
        r[c] = t[c] / torch.sin(t[c])

        return r

    @staticmethod
    def rsinc1_dt(t):
        """ d/dt(rsinc1) """
        e = 0.01
        r = torch.zeros_like(t)
        a = torch.abs(t)

        s = a < e
        c = (s == 0)
        t2 = t[s] ** 2
        r[s] = ((((127*t2)/30 + 31)*t2/28 + 7)*t2/30 + 1)*t[s]/3  # Taylor series O(t^8)
        r[c] = 1/torch.sin(t[c]) - (t[c]*torch.cos(t[c]))/(torch.sin(t[c])*torch.sin(t[c]))

        return r


    @staticmethod
    def rsinc1_dt_csc(t):
        """ d/dt(rsinc1) / sin(t) """
        e = 0.01
        r = torch.zeros_like(t)
        a = torch.abs(t)

        s = a < e
        c = (s == 0)
        t2 = t[s] ** 2
        r[s] = t2*(t2*((4*t2)/675 + 2/63) + 2/15) + 1/3  # Taylor series O(t^8)
        r[c] = (1/torch.sin(t[c]) - (t[c]*torch.cos(t[c]))/(torch.sin(t[c])*torch.sin(t[c]))) / torch.sin(t[c])

        return r


    @staticmethod
    def sinc2(t):
        """ sinc2: t -> (1 - cos(t)) / (t**2) """
        e = 0.01
        r = torch.zeros_like(t)
        a = torch.abs(t)

        s = a < e
        c = (s == 0)
        t2 = t ** 2
        r[s] = 1/2*(1-t2[s]/12*(1-t2[s]/30*(1-t2[s]/56)))  # Taylor series O(t^8)
        r[c] = (1-torch.cos(t[c]))/t2[c]

        return r


    @staticmethod
    def sinc2_dt(t):
        """ d/dt(sinc2) """
        e = 0.01
        r = torch.zeros_like(t)
        a = torch.abs(t)

        s = a < e
        c = (s == 0)
        t2 = t ** 2
        r[s] = -t[s]/12*(1 - t2[s]/5*(1.0/3 - t2[s]/56*(1.0/2 - t2[s]/135)))  # Taylor series O(t^8)
        r[c] = torch.sin(t[c])/t2[c] - 2*(1- torch.cos(t[c]))/(t2[c]*t[c])

        return r


    @staticmethod
    def sinc3(t):
        """ sinc3: t -> (t - sin(t)) / (t**3) """
        e = 0.01
        r = torch.zeros_like(t)
        a = torch.abs(t)

        s = a < e
        c = (s == 0)
        t2 = t[s] ** 2
        r[s] = 1/6*(1-t2/20*(1-t2/42*(1-t2/72)))  # Taylor series O(t^8)
        r[c] = (t[c]-torch.sin(t[c]))/(t[c]**3)

        return r

    @staticmethod
    def sinc3_dt(t):
        """ d/dt(sinc3) """
        e = 0.01
        r = torch.zeros_like(t)
        a = torch.abs(t)

        s = a < e
        c = (s == 0)
        t2 = t[s] ** 2
        r[s] = -t[s]/60*(1 - t2/21*(1 - t2/24*(1.0/2 - t2/165)))  # Taylor series O(t^8)
        r[c] = (3*torch.sin(t[c]) - t[c]*(torch.cos(t[c]) + 2))/(t[c]**4)

        return r


    @staticmethod
    def sinc4(t):
        """ sinc4: t -> 1/t^2 * (1/2 - sinc2(t))
                    = 1/t^2 * (1/2 - (1 - cos(t))/t^2)
        """
        e = 0.01
        r = torch.zeros_like(t)
        a = torch.abs(t)

        s = a < e
        c = (s == 0)
        t2 = t ** 2
        r[s] = 1/24*(1-t2/30*(1-t2/56*(1-t2/90)))  # Taylor series O(t^8)
        r[c] = (0.5 - (1 - torch.cos(t))/t2) / t2


class Sinc1_autograd(torch.autograd.Function):

    def __init__(self):
        self.sinc = Sinc()
    
    def forward(self, ctx, theta):
        ctx.save_for_backward(theta)
        return self.sinc.sinc1(theta)

 
    def backward(self,ctx, grad_output):
        theta, = ctx.saved_tensors
        grad_theta = None
        if ctx.needs_input_grad[0]:
            grad_theta = grad_output * self.sinc.sinc1_dt(theta).to(grad_output)
        return grad_theta

Sinc1 = Sinc1_autograd.apply

class RSinc1_autograd(torch.autograd.Function):
    def __init__(self):
        self.sinc = Sinc()

    def forward(self,ctx, theta):
        ctx.save_for_backward(theta)
        return self.sinc.rsinc1(theta)

    def backward(self, ctx, grad_output):
        theta, = ctx.saved_tensors
        grad_theta = None
        if ctx.needs_input_grad[0]:
            grad_theta = grad_output * self.rsinc1_dt(theta).to(grad_output)
        return grad_theta

RSinc1 = RSinc1_autograd.apply

class Sinc2_autograd(torch.autograd.Function):
    def __init__(self):
        self.sinc = Sinc()

    def forward(self, ctx, theta):
        ctx.save_for_backward(theta)
        return self.sinc.sinc2(theta)

    def backward(self,ctx, grad_output):
        theta, = ctx.saved_tensors
        grad_theta = None
        if ctx.needs_input_grad[0]:
            grad_theta = grad_output * self.sinc.sinc2_dt(theta).to(grad_output)
        return grad_theta

Sinc2 = Sinc2_autograd.apply

class Sinc3_autograd(torch.autograd.Function):
    def __init__(self):
        self.sinc = Sinc()

    def forward(self, ctx, theta):
        ctx.save_for_backward(theta)
        return self.sinc.sinc3(theta)

    def backward(self, ctx, grad_output):
        theta, = ctx.saved_tensors
        grad_theta = None
        if ctx.needs_input_grad[0]:
            grad_theta = grad_output * self.sinc.sinc3_dt(theta).to(grad_output)
        return grad_theta

Sinc3 = Sinc3_autograd.apply


class SO3(Sinc):
    def __init__(self) -> None:
        #self.sinc  = Sinc()
        super().__init__()

    @staticmethod
    def cross_prod(x, y):
        z = torch.cross(x.view(-1, 3), y.view(-1, 3), dim=1).view_as(x)
        return z

    def liebracket(self,x, y):
        return self.cross_prod(x, y)

    @staticmethod
    def mat(x):
        # size: [*, 3] -> [*, 3, 3]
        x_ = x.view(-1, 3)
        x1, x2, x3 = x_[:, 0], x_[:, 1], x_[:, 2]
        O = torch.zeros_like(x1)

        X = torch.stack((
            torch.stack((O, -x3, x2), dim=1),
            torch.stack((x3, O, -x1), dim=1),
            torch.stack((-x2, x1, O), dim=1)), dim=1)
        return X.view(*(x.size()[0:-1]), 3, 3)
    
    @staticmethod
    def vec(X):
        X_ = X.view(-1, 3, 3)
        x1, x2, x3 = X_[:, 2, 1], X_[:, 0, 2], X_[:, 1, 0]
        x = torch.stack((x1, x2, x3), dim=1)
        return x.view(*X.size()[0:-2], 3)

    @staticmethod
    def genvec():
        return torch.eye(3)

    def genmat(self):
        return self.mat(self.genvec())

    def RodriguesRotation(self, x):
        # for autograd
        w = x.view(-1, 3)
        t = w.norm(p=2, dim=1).view(-1, 1, 1)
        W = self.mat(w)
        S = W.bmm(W)
        I = torch.eye(3).to(w)

        # Rodrigues' rotation formula.
        #R = cos(t)*eye(3) + sinc1(t)*W + sinc2(t)*(w*w');
        #R = eye(3) + sinc1(t)*W + sinc2(t)*S

        R = I + self.sinc1(t)*W + self.sinc2(t)*S

        return R.view(*(x.size()[0:-1]), 3, 3)
    


    def exp(self, x:torch.Tensor) -> torch.Tensor:
        w = x.view(-1, 3)
        t = w.norm(p=2, dim=1).view(-1, 1, 1)
        W = self.mat(w)
        S = W.bmm(W)
        I = torch.eye(3).to(w)

        # Rodrigues' rotation formula.
        R = I + self.sinc1(t)*W + self.sinc2(t)*S

        return R.view(*(x.size()[0:-1]), 3, 3)
        
    @staticmethod
    def inverse(g):
        R = g.view(-1, 3, 3)
        Rt = R.transpose(1, 2)
        return Rt.view_as(g)

    @staticmethod
    def btrace(X):
        # batch-trace: [B, N, N] -> [B]
        n = X.size(-1)
        X_ = X.view(-1, n, n)
        tr = torch.zeros(X_.size(0)).to(X)
        for i in range(tr.size(0)):
            m = X_[i, :, :]
            tr[i] = torch.trace(m)
        return tr.view(*(X.size()[0:-2]))

    def log(self,g):
        eps = 1.0e-7
        R = g.view(-1, 3, 3)
        tr = self.btrace(R)
        c = (tr - 1) / 2
        t = torch.acos(c)
        sc = self.sinc1(t)
        idx0 = (torch.abs(sc) <= eps)
        idx1 = (torch.abs(sc) > eps)
        sc = sc.view(-1, 1, 1)

        X = torch.zeros_like(R)
        if idx1.any():
            X[idx1] = (R[idx1] - R[idx1].transpose(1, 2)) / (2*sc[idx1])

        if idx0.any():
            # t[idx0] == math.pi
            t2 = t[idx0] ** 2
            A = (R[idx0] + torch.eye(3).type_as(R).unsqueeze(0)) * t2.view(-1, 1, 1) / 2
            aw1 = torch.sqrt(A[:, 0, 0])
            aw2 = torch.sqrt(A[:, 1, 1])
            aw3 = torch.sqrt(A[:, 2, 2])
            sgn_3 = torch.sign(A[:, 0, 2])
            sgn_3[sgn_3 == 0] = 1
            sgn_23 = torch.sign(A[:, 1, 2])
            sgn_23[sgn_23 == 0] = 1
            sgn_2 = sgn_23 * sgn_3
            w1 = aw1
            w2 = aw2 * sgn_2
            w3 = aw3 * sgn_3
            w = torch.stack((w1, w2, w3), dim=-1)
            W = self.mat(w)
            X[idx0] = W

        x = self.vec(X.view_as(g))
        return x

    @staticmethod
    def transform(g:torch.Tensor, a:torch.Tensor):
        # g in SO(3):  * x 3 x 3
        # a in R^3:    * x 3[x N]
        if len(g.size()) == len(a.size()):
            b = g.matmul(a)
        else:
            b = g.matmul(a.unsqueeze(-1)).squeeze(-1)
        return b

    @staticmethod
    def group_prod(g, h):
        # g, h : SO(3)
        g1 = g.matmul(h)
        return g1



    def vecs_Xg_ig(self, x):
        """ Vi = vec(dg/dxi * inv(g)), where g = exp(x)
            (== [Ad(exp(x))] * vecs_ig_Xg(x))
        """
        t = x.view(-1, 3).norm(p=2, dim=1).view(-1, 1, 1)
        X = mat(x)
        S = X.bmm(X)
        #B = x.view(-1,3,1).bmm(x.view(-1,1,3))  # B = x*x'
        I = torch.eye(3).to(X)

        #V = sinc1(t)*eye(3) + sinc2(t)*X + sinc3(t)*B
        #V = eye(3) + sinc2(t)*X + sinc3(t)*S

        V = I + self.sinc2(t)*X + self.sinc3(t)*S

        return V.view(*(x.size()[0:-1]), 3, 3)
    

    def inv_vecs_Xg_ig(self,x):
        """ H = inv(vecs_Xg_ig(x)) """
        t = x.view(-1, 3).norm(p=2, dim=1).view(-1, 1, 1)
        X = self.mat(x)
        S = X.bmm(X)
        I = torch.eye(3).to(x)

        e = 0.01
        eta = torch.zeros_like(t)
        s = (t < e)
        c = (s == 0)
        t2 = t[s] ** 2
        eta[s] = ((t2/40 + 1)*t2/42 + 1)*t2/720 + 1/12 # O(t**8)
        eta[c] = (1 - (t[c]/2) / torch.tan(t[c]/2)) / (t[c]**2)

        H = I - 1/2*X + eta*S
        return H.view(*(x.size()[0:-1]), 3, 3)


class ExpMapSO3(torch.autograd.Function):

    def __init__(self, so3):
        self.so3 = SO3(Sinc())

 
    
    def forward(self, ctx, x):
        """ Exp: R^3 -> M(3),
            size: [B, 3] -> [B, 3, 3],
              or  [B, 1, 3] -> [B, 1, 3, 3]
        """
        ctx.save_for_backward(x)
        g = self.exp(x)
        return g


    def backward(self, ctx, grad_output):
        x, = ctx.saved_tensors
        g = self.so3.exp(x)
        gen_k = self.so3.genmat().to(x)
        #gen_1 = gen_k[0, :, :]
        #gen_2 = gen_k[1, :, :]
        #gen_3 = gen_k[2, :, :]

        # Let z = f(g) = f(exp(x))
        # dz = df/dgij * dgij/dxk * dxk
        #    = df/dgij * (d/dxk)[exp(x)]_ij * dxk
        #    = df/dgij * [gen_k*g]_ij * dxk

        dg = gen_k.matmul(g.view(-1, 1, 3, 3))
        # (k, i, j)
        dg = dg.to(grad_output)

        go = grad_output.contiguous().view(-1, 1, 3, 3)
        dd = go * dg
        grad_input = dd.sum(-1).sum(-1)

        return grad_input

Exp = ExpMapSO3.apply

class SE3(Sinc):
    """ 3-d rigid body transfomation group and corresponding Lie algebra. """

    def __init__(self) -> None:
        super().__init__()
        self.so3 = SO3()


    def twist_prod(self, x:torch.Tensor, y:torch.Tensor):
        x_ = x.view(-1, 6)
        y_ = y.view(-1, 6)

        xw, xv = x_[:, 0:3], x_[:, 3:6]
        yw, yv = y_[:, 0:3], y_[:, 3:6]

        zw = self.so3.cross_prod(xw, yw)
        zv = self.so3.cross_prod(xw, yv) + self.so3.cross_prod(xv, yw)

        z = torch.cat((zw, zv), dim=1)

        return z.view_as(x)

    def liebracket(self, x, y):
        return self.twist_prod(x, y)

    @staticmethod
    def mat(x:torch.Tensor):
        # size: [*, 6] -> [*, 4, 4]
        x_ = x.view(-1, 6)
        w1, w2, w3 = x_[:, 0], x_[:, 1], x_[:, 2]
        v1, v2, v3 = x_[:, 3], x_[:, 4], x_[:, 5]
        O = torch.zeros_like(w1)

        X = torch.stack((
            torch.stack((  O, -w3,  w2, v1), dim=1),
            torch.stack(( w3,   O, -w1, v2), dim=1),
            torch.stack((-w2,  w1,   O, v3), dim=1),
            torch.stack((  O,   O,   O,  O), dim=1)), dim=1)
        return X.view(*(x.size()[0:-1]), 4, 4)

    @staticmethod
    def vec(X:torch.Tensor):
        X_ = X.view(-1, 4, 4)
        w1, w2, w3 = X_[:, 2, 1], X_[:, 0, 2], X_[:, 1, 0]
        v1, v2, v3 = X_[:, 0, 3], X_[:, 1, 3], X_[:, 2, 3]
        x = torch.stack((w1, w2, w3, v1, v2, v3), dim=1)
        return x.view(*X.size()[0:-2], 6)

    @staticmethod
    def genvec():
        return torch.eye(6)

    
    def genmat(self):
        return self.mat(self.genvec())

    def exp(self, x:torch.Tensor):
        x_ = x.view(-1, 6)
        w, v = x_[:, 0:3], x_[:, 3:6]
        t = w.norm(p=2, dim=1).view(-1, 1, 1)
        W = self.so3.mat(w)
        S = W.bmm(W)
        I = torch.eye(3).to(w)

        # Rodrigues' rotation formula.
        #R = cos(t)*eye(3) + sinc1(t)*W + sinc2(t)*(w*w');
        #  = eye(3) + sinc1(t)*W + sinc2(t)*S
        R = I + self.sinc1(t)*W + self.sinc2(t)*S

        #V = sinc1(t)*eye(3) + sinc2(t)*W + sinc3(t)*(w*w')
        #  = eye(3) + sinc2(t)*W + sinc3(t)*S
        V = I + self.sinc2(t)*W + self.sinc3(t)*S

        p = V.bmm(v.contiguous().view(-1, 3, 1))

        z = torch.Tensor([0, 0, 0, 1]).view(1, 1, 4).repeat(x_.size(0), 1, 1).to(x)
        Rp = torch.cat((R, p), dim=2)
        g = torch.cat((Rp, z), dim=1)

        return g.view(*(x.size()[0:-1]), 4, 4)

    @staticmethod
    def inverse(g:torch.Tensor):
        g_ = g.view(-1, 4, 4)
        R = g_[:, 0:3, 0:3]
        p = g_[:, 0:3, 3]
        Q = R.transpose(1, 2)
        q = -Q.matmul(p.unsqueeze(-1))

        z = torch.Tensor([0, 0, 0, 1]).view(1, 1, 4).repeat(g_.size(0), 1, 1).to(g)
        Qq = torch.cat((Q, q), dim=2)
        ig = torch.cat((Qq, z), dim=1)

        return ig.view(*(g.size()[0:-2]), 4, 4)

    
    def log(self,g:torch.Tensor):
        g_ = g.view(-1, 4, 4)
        R = g_[:, 0:3, 0:3]
        p = g_[:, 0:3, 3]

        w = self.so3.log(R)
        H = self.so3.inv_vecs_Xg_ig(w)
        v = H.bmm(p.contiguous().view(-1, 3, 1)).view(-1, 3)

        x = torch.cat((w, v), dim=1)
        return x.view(*(g.size()[0:-2]), 6)

    @staticmethod
    def transform(g: torch.Tensor, a: torch.Tensor):
        # g : SE(3),  * x 4 x 4
        # a : R^3,    * x 3[x N]

        g_ = g.view(-1, 4, 4)
        R = g_[:, 0:3, 0:3].contiguous().view(*(g.size()[0:-2]), 3, 3)
        p = g_[:, 0:3, 3].contiguous().view(*(g.size()[0:-2]), 3)
        if len(g.size()) == len(a.size()):
            b = R.matmul(a) + p.unsqueeze(-1)
        else:
            b = R.matmul(a.unsqueeze(-1)).squeeze(-1) + p
        return b

    @staticmethod
    def rot_transform(g: torch.Tensor,a: torch.Tensor):
        """transform pcd in rotation component

        Args:
            g (B,K,4,4): SE3
            a (B,K,N,3): pcd
        """
        g_ = g[...,:3,:3]   # [B,K,3,3]
        return g_.matmul(a.transpose(2,3)).transpose(2,3)  # [B,K,3,3] x [B,K,3,N] -> [B,3,3,N] -> [B,3,N,3]

    @staticmethod
    def tsl_transform(g: torch.Tensor,a: torch.Tensor):
        """transform pcd in translation component

        Args:
            g (B,K,1,3): SE3
            a (B,K,N,3): pcd
        """
        return a + g  # (B,K,N,3) + (B,K,1,3) -> (B,K,N,3)

    @staticmethod
    def group_prod(g, h):
        # g, h : SE(3)
        g1 = g.matmul(h)
        return g1


class ExpMapSE3(torch.autograd.Function):
    """ Exp: se(3) -> SE(3)
    """

    def  __init__(self):
        self.se3 = SE3()

   
    def forward(self,ctx, x):
        """ Exp: R^6 -> M(4),
            size: [B, 6] -> [B, 4, 4],
              or  [B, 1, 6] -> [B, 1, 4, 4]
        """
        ctx.save_for_backward(x)
        g = self.se3.exp(x)
        return g

    def backward(self,ctx, grad_output):
        x, = ctx.saved_tensors
        g = self.se3.exp(x)
        gen_k = self.se3.genmat().to(x)

        # Let z = f(g) = f(exp(x))
        # dz = df/dgij * dgij/dxk * dxk
        #    = df/dgij * (d/dxk)[exp(x)]_ij * dxk
        #    = df/dgij * [gen_k*g]_ij * dxk

        dg = gen_k.matmul(g.view(-1, 1, 4, 4))
        # (k, i, j)
        dg = dg.to(grad_output)

        go = grad_output.contiguous().view(-1, 1, 4, 4)
        dd = go * dg
        grad_input = dd.sum(-1).sum(-1)

        return grad_input

Exp = ExpMapSE3.apply
# #EOF