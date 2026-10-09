import os
from pathlib import Path
import torch
from signalforge.runtime import gpu_lease
from signalforge.neural import require_cuda
from signalforge.vectorized_oracle import independent_linear_models,vectorized_linear_step

def test_vectorized_seeds_match_serial_rng_optimizer_and_independent_stopping():
    require_cuda()
    with gpu_lease(Path(os.environ['SIGNALFORGE_RUNTIME'])):
        seeds=[11,37,71];a,oa=independent_linear_models(seeds,3,'cuda');b,ob=independent_linear_models(seeds,3,'cuda')
        x=torch.arange(18,device='cuda',dtype=torch.float64).reshape(6,3)/18;y=x[:,:1]*.2
        assert not torch.equal(a[0].weight,a[1].weight)
        for active in [[True,True,True],[True,False,True],[True,False,True]]:
            prediction,_=vectorized_linear_step(a,oa,x,y,active)
            for i,(model,optimizer,enabled) in enumerate(zip(b,ob,active)):
                torch.testing.assert_close(prediction[i],model(x),rtol=1e-12,atol=1e-12)
                if enabled:
                    optimizer.zero_grad(set_to_none=True);((model(x)-y)**2).mean().backward();optimizer.step()
                for pa,pb in zip(a[i].parameters(),model.parameters()):
                    torch.testing.assert_close(pa,pb,rtol=1e-12,atol=1e-12)
                    for key in ['step','exp_avg','exp_avg_sq']:torch.testing.assert_close(oa[i].state[pa][key],optimizer.state[pb][key],rtol=1e-12,atol=1e-12)
        assert int(oa[0].state[a[0].weight]['step'])==3 and int(oa[1].state[a[1].weight]['step'])==1
