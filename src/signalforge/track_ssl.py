"""Train-cutoff SSL initialization shared across fixed downstream LR trials."""
import io,json
from .runtime import digest,file_hash,validate_bundle,commit_bundle
from .ssl import validate_pretrain,pretrain_gru


def frozen_ssl_initialization(repo,runtime,prepared,width,seed):
    training=prepared['training'];cutoff=prepared['transform'].fit_cutoff
    x,data_id=validate_pretrain(prepared['tx_sequence'],training.decision_time,training.max_dependency_available_at,cutoff)
    recipe={'data_id':data_id,'width':width,'seed':seed,'epochs':100,'method':'train_only_next_context_feature',
        'implementation':file_hash(repo/'src/signalforge/ssl.py')}
    identity=digest(recipe);directory=runtime/'artifacts/track_ssl'/identity
    import torch
    from .neural import require_cuda
    require_cuda()
    if (directory/'receipt.json').exists():
        validate_bundle(directory)
        if json.loads((directory/'recipe.json').read_text())!=recipe:raise PermissionError('SSL initialization recipe mismatch')
        state=torch.load(directory/'encoder.pt',map_location='cpu',weights_only=True)
        metadata=json.loads((directory/'metadata.json').read_text())
    else:
        result=pretrain_gru(x,training.decision_time,training.max_dependency_available_at,cutoff,width=width,epochs=100,seed=seed)
        state=result.pop('encoder_state');metadata=result
        stream=io.BytesIO();torch.save(state,stream)
        commit_bundle(directory,{'encoder.pt':stream.getvalue(),'recipe.json':recipe,'metadata.json':metadata},
            {'pretrain_data_id':data_id,'initialization_id':identity,'cutoff':cutoff,'reserved_access':False})
    return state,identity,metadata
