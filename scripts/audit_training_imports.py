from signalforge.runtime import paths,atomic_json
from signalforge.boundary import training_import_boundary
repo,runtime=paths();result=training_import_boundary(repo/'src/signalforge',['development','models','features','ssl','residual','track_engine','track_neural','track_inputs','capacity_development','fixed_gate'])
atomic_json(repo/'reports/training_import_boundary.json',result);print(result['state'],len(result['modules']))
