"""Optional correctness oracle, not an adopted research training backend."""
import torch


def independent_linear_models(seeds,features,device):
    if len(set(seeds))!=len(seeds):raise ValueError('Independent seed IDs required')
    models=[]
    for seed in seeds:
        with torch.random.fork_rng(devices=[]):model=torch.nn.Linear(features,1,dtype=torch.float64)
        generator=torch.Generator(device='cpu').manual_seed(seed)
        with torch.no_grad():
            for parameter in model.parameters():parameter.copy_(torch.randn(parameter.shape,generator=generator,dtype=torch.float64)*.1)
        models.append(model.to(device))
    return models,[torch.optim.AdamW(m.parameters(),lr=.001) for m in models]


def vectorized_linear_step(models,optimizers,x,y,active):
    if len(models)!=len(optimizers) or len(active)!=len(models) or len({id(m) for m in models})!=len(models) or len({id(o) for o in optimizers})!=len(models):
        raise ValueError('Independent model/optimizer ownership required')
    for model,optimizer in zip(models,optimizers):
        if {id(p) for p in model.parameters()}!={id(p) for g in optimizer.param_groups for p in g['params']}:
            raise ValueError('Optimizer belongs to another model')
        optimizer.zero_grad(set_to_none=True)
    parameters={name:torch.stack([dict(m.named_parameters())[name] for m in models]) for name,_ in models[0].named_parameters()}
    prediction=torch.vmap(lambda p:torch.func.functional_call(models[0],p,(x,)))(parameters)
    loss=((prediction-y)**2).mean(dim=(1,2))
    enabled=torch.as_tensor(active,dtype=torch.bool,device=x.device)
    if enabled.any():
        loss[enabled].sum().backward()
        for optimizer,enabled_seed in zip(optimizers,active):
            if enabled_seed:optimizer.step()
    return prediction.detach(),loss.detach()
