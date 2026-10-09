"""Static training import-graph audit; reserved loader is an orchestration boundary."""
import ast
from pathlib import Path
from .runtime import digest,file_hash


def training_import_boundary(package,roots):
    package=Path(package);pending=list(roots);visited={}
    while pending:
        module=pending.pop()
        if module in {'final'}:raise PermissionError('Training import graph reaches reserved target loader')
        if module in visited:continue
        path=package/(module.replace('.','/')+'.py')
        if not path.is_file():raise ValueError('Unknown local training module: '+module)
        tree=ast.parse(path.read_text());local=set()
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):
                if node.level:
                    if node.level!=1:raise ValueError('Unexpected relative training import')
                    if node.module:local.add(node.module)
                    else:local.update(a.name for a in node.names)
                elif node.module=='signalforge':local.update(a.name for a in node.names)
                elif node.module and node.module.startswith('signalforge.'):local.add(node.module.removeprefix('signalforge.'))
            elif isinstance(node,ast.Import):
                local.update(a.name.removeprefix('signalforge.') for a in node.names if a.name.startswith('signalforge.'))
            elif isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='import_module':
                if node.args and isinstance(node.args[0],ast.Constant) and str(node.args[0].value).startswith('signalforge.'):
                    local.add(str(node.args[0].value).removeprefix('signalforge.'))
        local={name for name in local if (package/(name.replace('.','/')+'.py')).is_file() or name=='final'}
        visited[module]={'file_sha256':file_hash(path),'local_imports':sorted(local)};pending+=sorted(local)
    return {'state':'PASSED_STATIC_TRAINING_IMPORT_BOUNDARY','roots':roots,'modules':visited,'graph_id':digest(visited),
            'claim_boundary':'Static repository import graph plus separately tested runtime scientific gate; not an arbitrary-code security sandbox',
            'reserved_access':False}
