from signalforge.capacity_development import run_capacity_development
from signalforge.runtime import paths

repo, runtime = paths()
result = run_capacity_development(repo, runtime)
print(result["state"], len(result["results"]))
