from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit

from model_arch import Adapter, WideResNetWithAdapter, WideResNetWithDropout
from cotta import get_tta_transforms

'''
this version model is all in train mode and use test batch stats
'''

# def get_max_entropy(n_classes):
#     prob = torch.full((n_classes,), 1.0 / n_classes)
#     e_max = -torch.sum(prob * torch.log(prob))
#     return e_max

def dropout_inference(x, model, n_iter=5, dropout_rate=0.4): #  n_iter可选5/10
        with torch.no_grad():
            # source model inference w/o dropout
            outputs = model(x, dropout_rate=0.0) # (batch_size, n_classes
            curr_pred = torch.argmax(outputs, dim=1) # batch_size
            # curr_conf = torch.max(F.softmax(outputs, dim=1), dim=1)[0] # batch_size

            # Dropout inference sampling
            x_expanded = x.repeat(n_iter, 1, 1, 1) # n_iter * batch_size, ...
            predictions = model(x_expanded, dropout_rate=dropout_rate) # n_iter * batch_size, n_classes
            predictions = F.softmax(predictions, dim=-1) 
            predictions = predictions.contiguous().view(n_iter, x.shape[0], -1) # n_iter, batch_size, n_classes
                    
            total_avg = torch.mean(predictions, dim=(0,1)) # n_classes 
            n_classes = total_avg.shape[-1]
            e_avg = (-total_avg * torch.log(total_avg + 1e-6)).sum()
            e_max = torch.log(torch.tensor(float(n_classes))) # log(n_classes)

            # Prediction disagreement with dropouts
            pred = torch.argmax(predictions, dim=2).permute(1, 0) # batch_size, n_iter 批次样本在每个iter的预测标记（索引     
            err = (curr_pred.unsqueeze(dim=1).repeat(1, n_iter) != pred).float().mean(dtype=torch.float64)
        
        return err.item(), e_avg.item(), e_max.item(), n_classes


def err_estimation(x, model, n_iter, dropout_rate):
    err, e_avg, e_max, n_classes = dropout_inference(x, model, n_iter, dropout_rate)
    est_err = err / (e_avg / e_max) ** 3 # AETTA
    est_err = max(0., min(1. - 1. / n_classes, est_err))
    print("err: ", err, "e_avg: ", e_avg, "est_err: ", est_err)
    return est_err
    


class PACE(nn.Module):
    
    def __init__(self, model_adapter, model_dropout, optimizer, steps=1, episodic=False, dropout_rate=0.4, n_iter=5):
        super().__init__()
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "pace requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # Create model_adapter and model_dropout using deepcopy to avoid unintended parameter sharing
        self.model_adapter = model_adapter
        self.model_dropout = model_dropout
        
        self.transform = get_tta_transforms()
        self.n_iter = n_iter
        self.dropout_rate = dropout_rate

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_adapter_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model_adapter, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        # 在这里做dropout inference，得到batch err/acc估计，然后用这个作为adapter_ratio
        with torch.no_grad():
            err_est = err_estimation(x, self.model_dropout, self.n_iter, self.dropout_rate)      
            
            adapter_ratio = err_est

        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self.model_adapter, self.optimizer, self.transform, adapter_ratio)

        return outputs

    def reset(self):
        if self.model_adapter_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model_adapter, self.optimizer,
                                 self.model_adapter_state, self.optimizer_state)

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer, transform, adapter_ratio):
    """Forward and adapt model on batch of data.
    """
    # forward
    outputs = model(x, adapter_ratio) # adapter_ratio * adapter_out + (1 - adapter_ratio) * out
    outputs_aug = model(transform(x), adapter_ratio=1.0) # adapter_out
    
    # adapt
    optimizer.zero_grad()
    loss = (softmax_entropy(outputs, outputs_aug)).mean(0) + (softmax_entropy(outputs, outputs)).mean(0)
    loss.backward()
    optimizer.step()
    return outputs


@torch.jit.script
def softmax_entropy(x, x_aug):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x_aug.log_softmax(1)).sum(1)


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def collect_params(model):
    """Collect all trainable parameters.

    Walk the model's modules and collect all parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, Adapter):
            for np, p in m.named_parameters():
                if ('weight' in np or 'bias' in np) and p.requires_grad:  
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def configure_model_adapter(model, reduction):
    model_adapter = WideResNetWithAdapter(reduction=reduction)
    model_adapter.load_state_dict(model.state_dict(), strict=False)
    model_adapter.train()
    # disable grad, to (re-)enable only what we update
    model_adapter.requires_grad_(False)
    for m in model_adapter.modules():
        # Ensure BatchNorm layers use train-time global statistics
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None  
        if isinstance(m, Adapter):
            m.requires_grad_(True)
    return model_adapter


def configure_model_dropout(model):
    model_dropout = WideResNetWithDropout()
    model_dropout.load_state_dict(model.state_dict(), strict=False)
    model_dropout.train() # for dropout inference
    model_dropout.requires_grad_(False)
    for m in model_dropout.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None  
    return model_dropout


def check_model(model):
    is_training = model.training
    assert is_training, "pace needs train mode: call model.train()"
    
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "pace needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "pace should not update all params: " \
                               "check which require grad"
    
    for nm, m in model.named_modules():
        if nm == '': # 根模块
            continue
        elif 'adapter' in nm: # 检查adapter模块及其子模块
            adapter_param_grads = [p.requires_grad for p in m.parameters()]
            assert all(adapter_param_grads), f"Adapter {nm} parameters should have requires_grad=True"
        # Ensure BatchNorm layers are in eval mode
        else:
            module_param_grads = [p.requires_grad for p in m.parameters()]
            assert not any(module_param_grads), f"Non-Adapter module {nm} should not have requires_grad=True"
    
    has_adapter = any([isinstance(m, Adapter) for m in model.modules()])
    assert has_adapter, "pace needs Adapter for its optimization"


