#Config Reader 
import yaml
from dataclasses import dataclass
from pathlib import Path


@dataclass
class transformer_config:
    name: str
    model_class : str
    layers: int
    context_length: int
    embedding_dim: int
    vocab_size: int
    dropout: float
    attention_heads:int
    qkv_bias:bool 

@dataclass
class optimizer_config: 
    learning_rate: str
    batch_size: int
    epochs: int
    optimizer: str

@dataclass 
class configuration:
    model: transformer_config
    optimizer: optimizer_config


BASE_CONFIG_DIR = Path(__file__).parent / "configs"

def config(filename:str):
    file_path = (BASE_CONFIG_DIR / filename).resolve()

    if not str(file_path).startswith(str(BASE_CONFIG_DIR.resolve())):
        raise ValueError(f"Access denied: {filename} is outside the configs folder")

    if not file_path.is_file():
        raise FileNotFoundError(f"Config file not found: {filename}")
    
    with open(file_path, "r") as f:
        raw_cfg = yaml.safe_load(f)
        model = transformer_config(**raw_cfg["model"])
        optimizer = optimizer_config(**raw_cfg["optimizer"])
        return configuration(model=model, optimizer=optimizer)

