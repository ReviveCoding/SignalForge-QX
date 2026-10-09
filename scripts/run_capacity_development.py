from signalforge.runtime import paths
from signalforge.capacity_development import run_capacity_development
repo,runtime=paths();result=run_capacity_development(repo,runtime)
print(result['state'],len(result['results']))
