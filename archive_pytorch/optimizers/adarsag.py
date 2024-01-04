import torch
import warnings

class AdaRSAG(torch.optim.Optimizer):
    r"""
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (lambda) (required)
        kappa (float): lambda  (default: 1000)
        xi (float, optional): statistical advantage parameter (default: 10)
        smallConst (float, optional): any value <=1 (default: 0.7)
    Example:
        >>> from RSAG import *
        >>> optimizer = RSAG(model.parameters(), lr=0.1, kappa = 1000.0, xi = 10.0)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(self, 
                 params, 
                 lr=0.01, 
                 alpha = 0.1, 
                 beta = 0.1): #, smallConst = 0.7, weight_decay=0):
        #defaults = dict(lr=lr, kappa=kappa, xi, smallConst=smallConst,
                        # weight_decay=weight_decay)
        
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("Invalid alpha: {}".format(alpha))
        if beta < 0.0:
            raise ValueError("Invalid beta: {}".format(beta))
        
        defaults = dict(lr=lr, alpha=alpha, beta=beta)
        super(AdaRSAG, self).__init__(params, defaults)

        self._steps = 0
        self.state['step_sizes'] = []


    # def __setstate__(self, state):
    #     super(RSAG, self).__setstate__(state)

    def step(self, closure=None):
        """ Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        self._steps += 1

        
        for group in self.param_groups:
            # weight_decay = group['weight_decay']
            lr = group['lr']
            alpha, beta = group['alpha'], group['beta']
            alpha_bar = 1.0-alpha
            momentum_buffer_list = []

            # Update lr/adaptive step size


            # UPDATE GROUPS
            for p in group['params']:

                if p.grad is None:
                    continue

                d_w = p.grad.data
                # w = p.data
                param_state = self.state[p]

                # if weight_decay != 0:
                #     grad_d.add_(weight_decay, p.data)
                if self._steps == 1:
                    param_state['momentum_aggr'] =  p.data.detach().clone()
                    param_state['prev_momentum_aggr'] = torch.zeros_like(p.data)
                
                buf = param_state['momentum_aggr'].detach()
                aggr_grad = (buf-param_state['prev_momentum_aggr'])
                aggr_grad.mul_(alpha_bar)
                aggr_grad.add_(d_w, alpha=alpha)
                
                param_state['prev_momentum_aggr'] = buf.clone()
                
                # Update momentum buffer:'
                buf.mul_(alpha_bar)
                buf.add_(p.data, alpha=alpha)
                buf.add_(aggr_grad, alpha=-beta)
                
                p.data.add_(aggr_grad, alpha=-lr)
                # print('aggr_grad', aggr_grad)
            
            # UPDATE MOMENTUM BUFFER
            # for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
            #     state = self.state[p]
                
            #     state['momentum_buffer'] = momentum_buffer

        return loss