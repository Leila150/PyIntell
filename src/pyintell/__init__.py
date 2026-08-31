"""pyintell: a modular framework for building and running AI models."""

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
from .permissions import ToolPermissions, ToolPermissionError, ToolConfirmationRequired
from .tools import Tool, ToolRegistry, tool, code_execution
from .packages import PackageInfo, TRUSTED_PACKAGES, register_package, get_package, list_packages, package_installed
from .knowledge import Memory, KnowledgeStore, MemoryStore
from .dataset import Dataset, load_dataset, save_dataset
from .verification import verify_url, is_trusted_domain, rank_sources, fingerprint
from .source_verification import Evidence, VerificationReport, verify_sources
from .evaluation import EvaluationResult, Evaluator, evaluate
from .crashlog import CrashLogger, get_logger, log, crash
from .terminal import Terminal, TerminalResult, TerminalDisabledError, TerminalUnavailableError, terminal
from .bash import Bash, bash
from .languages import Language, LANGUAGES, GUI_FRAMEWORKS, register_language, register, get_language, detect_language, list_languages
from .execution import CodeExecutor, ExecutionPolicy, ExecutionResult, ExecutionDisabledError, RuntimeUnavailableError, executor, code_execute, run_code
from .capabilities import Config, EventBus, LRUCache, ScopedMemory, Context, TaskPlan, TaskStep, RetryPolicy, config, events, cache, memory, context, fingerprint as capability_fingerprint
from .model import Model
from .builder import build

__version__ = "0.2.0"

save_model = save_model
load_model = load_model
generate = generate
model_run = model_run

__all__ = [name for name in globals() if not name.startswith("_")]
