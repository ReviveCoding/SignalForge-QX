"""Retrained fixed valid-expert gate; separate from frozen input interventions."""
from .track_neural import ExplicitSourceCUDA


class FixedGateCUDA(ExplicitSourceCUDA):
    def build(self,features):
        import torch
        model=super().build(features)
        with torch.no_grad():model.gate.weight.zero_();model.gate.bias.zero_()
        model.gate.weight.requires_grad_(False);model.gate.bias.requires_grad_(False)
        return model

    @staticmethod
    def load(path):
        import torch
        from .neural import require_cuda
        require_cuda();state=torch.load(path,map_location='cuda',weights_only=False)
        config=dict(state['config']);mask_id=config.pop('raw_source_mask_id',None)
        adapter=FixedGateCUDA(**config)
        if mask_id:adapter.config['raw_source_mask_id']=mask_id
        adapter.features=state['features'];adapter.model=adapter.build(adapter.features).cuda()
        adapter.model.load_state_dict(state['model']);adapter.model.eval()
        adapter.losses=state['losses'];adapter.validation_losses=state['validation_losses']
        adapter.initialization_id=state.get('initialization_id')
        if torch.count_nonzero(adapter.model.gate.weight) or torch.count_nonzero(adapter.model.gate.bias):
            raise ValueError('Stored fixed gate is not uniform')
        return adapter
