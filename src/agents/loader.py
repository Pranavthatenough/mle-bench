import importlib, importlib.util, inspect, sys
from pathlib import Path
from typing import Dict, List, Optional, Type, Union
from src.agents.base import BaseAgent

AGENT_REGISTRY: Dict[str, Type[BaseAgent]] = {}

def register_agent(name_or_cls: Union[str, Type[BaseAgent]]):
    if isinstance(name_or_cls, str):
        def decorator(cls: Type[BaseAgent]):
            AGENT_REGISTRY[name_or_cls.lower().strip()] = cls
            return cls
        return decorator
    elif inspect.isclass(name_or_cls) and issubclass(name_or_cls, BaseAgent):
        AGENT_REGISTRY[name_or_cls.__name__.lower()] = name_or_cls
        return name_or_cls

def discover_agents_in_dir(directory: Union[str, Path]):
    dir_path = Path(directory).resolve()
    if not dir_path.exists(): return
    for py_file in dir_path.glob("*.py"):
        if py_file.name.startswith("__"): continue
        try:
            spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                for _, cls in inspect.getmembers(mod, inspect.isclass):
                    if issubclass(cls, BaseAgent) and cls is not BaseAgent:
                        AGENT_REGISTRY[cls.__name__.lower()] = cls
        except Exception: pass

def load_agent(agent_spec: str, **kwargs) -> BaseAgent:
    discover_agents_in_dir(Path(__file__).parent)
    key = agent_spec.strip().lower().replace("-", "_")
    if key in AGENT_REGISTRY:
        return AGENT_REGISTRY[key](**kwargs)
    for k, cls in AGENT_REGISTRY.items():
        if key in k:
            return cls(**kwargs)
    raise ValueError(f"Agent '{agent_spec}' not found. Available: {list(AGENT_REGISTRY.keys())}")