"""pyintell: a modular NumPy-based framework for building AI models."""

from .tokenization import *
from .embeddings import *
from .attention import *
from .layers import *
from .normalization import *
from .activations import *
from .positional import *
from .residual import *
from .transformer import *
from .loss import *
from .optim import *
from .training import *
from .generation import *
from .decoding import *
from .system import *
from .autograd import *
from .utilities import *
from .serialization import *
from .quantization import *
from .scheduling import *
from .finetuning import *
from .focus import SUPPORTED_FOCUSES, FOCUS_PROFILES, normalize_focus, build_focus_config, focus_description
from .inference import InferenceSession, inference, run
from .tools import Tool, ToolRegistry, tool
from .builtin_tools import web_search, pypi_search, pip, bash
from .packages import PackageInfo, TRUSTED_PACKAGES, register_package, get_package, list_packages, package_installed
from .knowledge import Memory, KnowledgeStore, MemoryStore
from .dataset import Dataset, load_dataset, save_dataset
from .verification import verify_url, is_trusted_domain, rank_sources, fingerprint
from .evaluation import EvaluationResult, Evaluator, evaluate
from .crashlog import CrashLogger, get_logger, log, crash
from .model import Model
from .builder import build

__version__ = "0.1.0"

save_model = save_model
load_model = load_model
generate = generate
model_run = model_run

__all__ = [name for name in globals() if not name.startswith("_")]
