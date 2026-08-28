"""pyintell: a modular NumPy-based framework for building AI models."""

from .tokenization import *
from .embeddings import *
from .attention import *
from .layers import *
from .transformer import *
from .loss import *
from .optim import *
from .training import *
from .generation import *
from .system import *
from .autograd import *
from .utilities import *
from .serialization import *
from .quantization import *
from .scheduling import *
from .finetuning import *
from .focus import SUPPORTED_FOCUSES, FOCUS_PROFILES, normalize_focus, build_focus_config, focus_description
from .model import Model
from .builder import build

__version__ = "0.1.0"

# Explicitly expose the high-level model-management API.
save_model = save_model
load_model = load_model
generate = generate
model_run = model_run

__all__ = [name for name in globals() if not name.startswith("_")]
