"""Flat-name archive adapter for immutable runtime bundle contract."""
from pathlib import Path
from signalforge.runtime import digest

def flatten(files):
    flat={};mapping={}
    for relative,value in files.items():
        name=digest({'relative_path':relative})[:16]+'_'+Path(relative).name
        if name in flat or Path(name).name!=name or name=='receipt.json':raise ValueError('Flat bundle identity collision')
        flat[name]=value;mapping[name]=relative
    return flat,mapping