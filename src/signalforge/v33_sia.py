"""Versioned retrospective SIA; leaves features.TrainTransform unchanged."""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from .runtime import digest
from .pit import _aware_utc

KINDS = {'numeric', 'release_age', 'reference_age', 'mask', 'coverage'}
VERSION = 'sgqx-v3.3-sia-log1p-day-constant-ood-v1'

@dataclass(frozen=True)
class Feature:
    name: str
    kind: str
    source: str

def checked_schema(schema):
    schema = tuple(schema)
    if not schema or any(not isinstance(s, Feature) or s.kind not in KINDS or
                         not s.name or not s.source for s in schema):
        raise ValueError('Explicit typed schema required')
    if len({s.name for s in schema}) != len(schema):
        raise ValueError('Duplicate feature name')
    return schema

def schema_from_specs(specs, feature_names):
    """Enumerate known fields from exact specs; never classify arbitrary substrings."""
    schema = []
    for spec in specs:
        name, source = spec['name'], spec['source']
        schema.extend([Feature(name, 'numeric', source),
                       Feature(name+'_release_age', 'release_age', source),
                       Feature(name+'_reference_age', 'reference_age', source),
                       Feature(name+'_observed', 'mask', source),
                       Feature(name+'_coverage', 'coverage', source)])
        if spec.get('percentile_window'):
            schema.append(Feature(name+'_past_percentile', 'numeric', source))
    schema.extend([Feature('season_sin', 'numeric', 'market'),
                   Feature('season_cos', 'numeric', 'market'),
                   Feature('context_padding_indicator', 'mask', 'market')])
    if [s.name for s in schema] != list(feature_names):
        raise ValueError('Exact schema/feature-name alignment required')
    return checked_schema(schema)

class SemanticAgeTransform:
    def __init__(self, schema, constant_tolerance=1e-8):
        self.schema = checked_schema(schema)
        if constant_tolerance != 1e-8:
            raise ValueError('Registered constant tolerance required')
        self.constant_tolerance = constant_tolerance

    def _array(self, x, names):
        a = np.asarray(x, dtype=float)
        if list(names) != [s.name for s in self.schema] or a.ndim != 2 or a.shape[1] != len(self.schema):
            raise ValueError('Exact named feature alignment required')
        if np.isinf(a).any():
            raise ValueError('Infinite input')
        for i, spec in enumerate(self.schema):
            seen = a[:, i][np.isfinite(a[:, i])]
            if spec.kind in {'release_age', 'reference_age'} and (seen < 0).any():
                raise ValueError('Negative observed age')
            if spec.kind in {'mask', 'coverage'} and (len(seen) != len(a) or
                    not np.isin(seen, [0., 1.]).all()):
                raise ValueError('Finite typed binary mask/coverage required')
        return a

    def fit(self, x, names, dates, cutoff, *, available_at):
        a = self._array(x, names)
        dates = pd.DatetimeIndex(_aware_utc(dates, 'fit origins'))
        available = pd.DatetimeIndex(_aware_utc(available_at, 'feature availability'))
        cut = pd.Timestamp(cutoff)
        if cut.tzinfo is None or not len(a) or len(dates) != len(a) or len(available) != len(a):
            raise ValueError('Aware aligned nonempty fit cohort required')
        if dates.max() >= pd.Timestamp('2024-01-01T00:00Z'):
            raise PermissionError('Reserved cohort forbidden')
        if (dates > cut).any() or (available > dates).any():
            raise PermissionError('Future fit or source dependency')
        self.center = np.zeros(a.shape[1]); self.scale = np.ones(a.shape[1])
        self.constant = np.zeros(a.shape[1], bool)
        self.unseen = ~np.isfinite(a).any(axis=0)
        for i, spec in enumerate(self.schema):
            if spec.kind != 'numeric':
                continue
            seen = a[:, i][np.isfinite(a[:, i])]
            self.center[i] = float(np.median(seen)) if len(seen) else 0.
            filled = np.where(np.isfinite(a[:, i]), a[:, i], self.center[i])
            sd = float(filled.std())
            self.constant[i] = sd <= self.constant_tolerance
            self.scale[i] = 1. if self.constant[i] else sd
        self.fit_cutoff = cut.isoformat()
        self.fit_rows_hash = digest({'dates': dates.astype(str).tolist(),
            'available_at': available.astype(str).tolist(),
            'values': np.where(np.isfinite(a), a, 0).tolist(),
            'observed': np.isfinite(a).tolist()})
        return self

    def transform(self, x, names):
        if not hasattr(self, 'center'):
            raise ValueError('Unfitted transformer')
        a = self._array(x, names); observed = np.isfinite(a)
        out = np.zeros_like(a); ood = np.zeros_like(a, dtype=bool)
        for i, spec in enumerate(self.schema):
            if spec.kind in {'release_age', 'reference_age'}:
                out[:, i] = np.log1p(np.where(observed[:, i], a[:, i], 0.))
            elif spec.kind in {'mask', 'coverage'}:
                out[:, i] = a[:, i]
            elif self.constant[i]:
                ood[:, i] = observed[:, i] & (self.unseen[i] |
                    (np.abs(a[:, i]-self.center[i]) > self.constant_tolerance))
            else:
                out[:, i] = (np.where(observed[:, i], a[:, i], self.center[i])-self.center[i])/self.scale[i]
        # OOD is a separate typed output, not a hidden clipping/learned test cap.
        return {'values': out, 'observed': observed, 'numeric_ood': ood,
                'model_matrix': np.concatenate([out, observed.astype(float), ood.astype(float)], axis=1)}

    def manifest(self):
        payload = {'version': VERSION, 'schema': [asdict(s) for s in self.schema],
            'age_unit_days': 1., 'constant_tolerance': self.constant_tolerance,
            'center': self.center.tolist(), 'scale': self.scale.tolist(),
            'constant': self.constant.tolist(), 'unseen': self.unseen.tolist(),
            'fit_cutoff': self.fit_cutoff, 'fit_rows_hash': self.fit_rows_hash,
            'output_blocks': ['values', 'observed', 'numeric_ood']}
        return {**payload, 'manifest_id': digest(payload)}

    @classmethod
    def from_manifest(cls, manifest):
        payload = {k:v for k,v in manifest.items() if k != 'manifest_id'}
        if digest(payload) != manifest.get('manifest_id') or payload.get('version') != VERSION:
            raise PermissionError('SIA manifest integrity/version mismatch')
        obj = cls([Feature(**v) for v in payload['schema']], payload['constant_tolerance'])
        for key in ['center', 'scale', 'constant', 'unseen']:
            setattr(obj, key, np.asarray(payload[key], dtype=bool if key in {'constant','unseen'} else float))
        if any(getattr(obj,k).shape != (len(obj.schema),) for k in ['center','scale','constant','unseen']):
            raise ValueError('Manifest dimensions')
        if not np.isfinite(obj.center).all() or not np.isfinite(obj.scale).all() or (obj.scale <= 0).any():
            raise ValueError('Manifest numerical state')
        obj.fit_cutoff=payload['fit_cutoff']; obj.fit_rows_hash=payload['fit_rows_hash']
        return obj

def prepare_sia(training, testing, contexts, cutoff, schema, *, neural_contract=None):
    """Pure v3.3 preprocessing integration; no model fit or artifact writes."""
    from .track_engine import asset_normalizer, per_date_weights
    training=training.sort_values(['decision_time','asset'])
    tx=np.asarray(contexts)[training.sequence_index]; vx=np.asarray(contexts)[testing.sequence_index]
    names=[s.name for s in schema]; steps=tx.shape[1]
    transform=SemanticAgeTransform(schema).fit(tx.reshape(-1,tx.shape[-1]), names,
        np.repeat(training.decision_time.to_numpy(),steps), cutoff,
        available_at=np.repeat(training.max_dependency_available_at.to_numpy(),steps))
    scales, normalizer_id=asset_normalizer(training,cutoff)
    if not set(testing.asset)<=set(scales): raise ValueError('Unknown evaluation asset')
    def apply(raw):
        result=transform.transform(raw.reshape(-1,raw.shape[-1]),names)
        return {k:v.reshape(len(raw),raw.shape[1],-1) for k,v in result.items()}
    result={'train':apply(tx), 'test':apply(vx), 'transform':transform,
            'asset_scales':scales,'normalizer_id':normalizer_id,
            'train_target':training.y.to_numpy()/training.asset.map(scales).to_numpy(),
            'test_scale':testing.asset.map(scales).to_numpy(),
            'weights':per_date_weights(training)}
    if neural_contract is not None:
        from .track_neural import raw_source_validity
        dimension=tx.shape[-1]
        def expanded(columns):
            if len(columns)!=len(set(columns)) or any(i<0 or i>=dimension for i in columns):
                raise ValueError('Typed neural column alignment')
            return [i+offset for offset in [0,dimension,2*dimension] for i in columns]
        result['neural_args']={'base_columns':expanded(neural_contract['base_raw_columns']),
            'source_columns':[expanded(v) for v in neural_contract['source_raw_columns']],
            'meta_columns':expanded(neural_contract['meta_raw_columns'])}
        result['source_valid']=raw_source_validity(tx,neural_contract['source_value_columns'])
        result['test_source_valid']=raw_source_validity(vx,neural_contract['source_value_columns'])
    return result