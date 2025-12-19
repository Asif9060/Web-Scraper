"""
Local LLM interface using Hugging Face Transformers.

Provides a memory-efficient implementation optimized for CPU-only
8GB RAM environments using Qwen2.5-1.5B-Instruct.
"""

import gc
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator, Literal

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from web_intel.config import Settings
from web_intel.core.exceptions import ModelLoadError, InferenceError
from web_intel.utils.logging import get_logger
from web_intel.utils.metrics import Metrics

logger = get_logger(__name__)


class Role(str, Enum):
    """Message roles in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ConversationMessage:
    """A message in a conversation."""

    role: Role
    content: str

    def to_dict(self) -> dict:
        """Convert to dictionary format for chat templates."""
        return {"role": self.role.value, "content": self.content}


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = False
    repetition_penalty: float = 1.1
    stop_sequences: list[str] = field(default_factory=list)

    def to_generate_kwargs(self) -> dict:
        """Convert to kwargs for model.generate()."""
        kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "do_sample": self.do_sample,
            "repetition_penalty": self.repetition_penalty,
        }

        # Temperature must be > 0 when sampling
        if self.do_sample and self.temperature == 0:
            kwargs["temperature"] = 0.01

        return kwargs


@dataclass
class GenerationResult:
    """Result of a generation request."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: Literal["stop", "length", "error"] = "stop"
    model_name: str = ""

    @property
    def tokens_per_second(self) -> float:
        """Placeholder for performance tracking."""
        return 0.0


class StopOnSequences(StoppingCriteria):
    """Custom stopping criteria for specific sequences."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        stop_sequences: list[str],
    ) -> None:
        self.tokenizer = tokenizer
        self.stop_sequences = stop_sequences
        self.stop_ids = [
            tokenizer.encode(seq, add_special_tokens=False)
            for seq in stop_sequences
        ]

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs,
    ) -> bool:
        """Check if any stop sequence has been generated."""
        # Get the last generated tokens
        for stop_ids in self.stop_ids:
            if len(stop_ids) == 0:
                continue

            seq_len = len(stop_ids)
            if input_ids.shape[1] >= seq_len:
                last_tokens = input_ids[0, -seq_len:].tolist()
                if last_tokens == stop_ids:
                    return True

        return False


class LocalLLM:
    """
    Local LLM interface using Hugging Face Transformers.

    Optimized for CPU-only, 8GB RAM environments with:
    - Lazy model loading to reduce startup time
    - Memory-efficient inference with low_cpu_mem_usage
    - Streaming support for responsive output
    - Automatic cleanup and garbage collection

    Example:
        >>> llm = LocalLLM.from_settings(settings)
        >>> result = llm.generate("Summarize this text: ...")
        >>> print(result.text)

        >>> # With conversation
        >>> messages = [
        ...     ConversationMessage(Role.SYSTEM, "You are a helpful assistant."),
        ...     ConversationMessage(Role.USER, "What is web scraping?"),
        ... ]
        >>> result = llm.chat(messages)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "cpu",
        torch_dtype: str = "float32",
        low_cpu_mem_usage: bool = True,
        trust_remote_code: bool = False,
        cache_dir: Path | None = None,
        default_config: GenerationConfig | None = None,
    ) -> None:
        """
        Initialize local LLM.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run on (cpu, cuda, mps)
            torch_dtype: PyTorch dtype for weights
            low_cpu_mem_usage: Use memory-efficient loading
            trust_remote_code: Trust remote code in model repo
            cache_dir: Directory to cache downloaded models
            default_config: Default generation configuration
        """
        self.model_name = model_name
        self.device = device
        self.torch_dtype = self._parse_dtype(torch_dtype)
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.trust_remote_code = trust_remote_code
        self.cache_dir = str(cache_dir) if cache_dir else None
        self.default_config = default_config or GenerationConfig()

        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._lock = threading.Lock()
        self._loaded = False

        logger.info(
            f"LocalLLM initialized (model={model_name}, device={device}, "
            f"dtype={torch_dtype})"
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> "LocalLLM":
        """
        Create LocalLLM from application settings.

        Args:
            settings: Application settings

        Returns:
            Configured LocalLLM instance
        """
        llm_settings = settings.local_llm

        default_config = GenerationConfig(
            max_new_tokens=llm_settings.max_new_tokens,
            temperature=llm_settings.temperature,
            top_p=llm_settings.top_p,
            do_sample=llm_settings.do_sample,
        )

        return cls(
            model_name=llm_settings.model_name,
            device=llm_settings.device,
            torch_dtype=llm_settings.torch_dtype,
            low_cpu_mem_usage=llm_settings.low_cpu_mem_usage,
            trust_remote_code=llm_settings.trust_remote_code,
            cache_dir=llm_settings.cache_dir,
            default_config=default_config,
        )

    def _parse_dtype(self, dtype_str: str) -> torch.dtype:
        """Parse dtype string to torch.dtype."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(dtype_str, torch.float32)

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def load(self) -> None:
        """
        Load model and tokenizer.

        Loads the model lazily on first use. Thread-safe.

        Raises:
            ModelLoadError: If model loading fails
        """
        if self._loaded:
            return

        with self._lock:
            # Double-check after acquiring lock
            if self._loaded:
                return

            logger.info(f"Loading model: {self.model_name}")

            try:
                # Load tokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=self.trust_remote_code,
                    cache_dir=self.cache_dir,
                )

                # Ensure pad token is set
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token

                # Load model with memory optimizations
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=self.torch_dtype,
                    low_cpu_mem_usage=self.low_cpu_mem_usage,
                    trust_remote_code=self.trust_remote_code,
                    cache_dir=self.cache_dir,
                    device_map=self.device if self.device != "cpu" else None,
                )

                # Move to device if CPU
                if self.device == "cpu":
                    self._model = self._model.to(self.device)

                # Set to evaluation mode
                self._model.eval()

                self._loaded = True
                logger.info(
                    f"Model loaded successfully "
                    f"(params={self._count_parameters()})"
                )

            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise ModelLoadError(
                    f"Failed to load model {self.model_name}",
                    details={"error": str(e), "model": self.model_name},
                ) from e

    def unload(self) -> None:
        """
        Unload model to free memory.

        Useful when switching between models or when memory is needed.
        """
        with self._lock:
            if not self._loaded:
                return

            logger.info("Unloading model")

            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            self._loaded = False

            # Force garbage collection
            gc.collect()

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Model unloaded")

    def _count_parameters(self) -> str:
        """Count and format model parameters."""
        if self._model is None:
            return "0"

        params = sum(p.numel() for p in self._model.parameters())

        if params >= 1e9:
            return f"{params / 1e9:.1f}B"
        elif params >= 1e6:
            return f"{params / 1e6:.1f}M"
        else:
            return f"{params / 1e3:.1f}K"

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        system_prompt: str | None = None,
    ) -> GenerationResult:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt text
            config: Generation configuration (uses default if None)
            system_prompt: Optional system prompt to prepend

        Returns:
            GenerationResult with generated text

        Raises:
            InferenceError: If generation fails
        """
        # Build messages
        messages = []
        if system_prompt:
            messages.append(ConversationMessage(Role.SYSTEM, system_prompt))
        messages.append(ConversationMessage(Role.USER, prompt))

        return self.chat(messages, config)

    def chat(
        self,
        messages: list[ConversationMessage],
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        """
        Generate a response for a conversation.

        Args:
            messages: Conversation history
            config: Generation configuration

        Returns:
            GenerationResult with assistant response

        Raises:
            InferenceError: If generation fails
        """
        self.load()
        config = config or self.default_config
        metrics = Metrics.get()
        start_time = time.perf_counter()

        try:
            # Format messages using chat template
            message_dicts = [m.to_dict() for m in messages]
            prompt_text = self._tokenizer.apply_chat_template(
                message_dicts,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Tokenize
            inputs = self._tokenizer(
                prompt_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            prompt_tokens = inputs["input_ids"].shape[1]

            # Build stopping criteria
            stopping_criteria = None
            if config.stop_sequences:
                stopping_criteria = StoppingCriteriaList([
                    StopOnSequences(self._tokenizer, config.stop_sequences)
                ])

            # Generate
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    **config.to_generate_kwargs(),
                    stopping_criteria=stopping_criteria,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )

            # Decode only new tokens
            new_tokens = outputs[0, prompt_tokens:]
            completion_tokens = len(new_tokens)
            generated_text = self._tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
            )

            # Trim stop sequences from output
            for seq in config.stop_sequences:
                if generated_text.endswith(seq):
                    generated_text = generated_text[: -len(seq)]

            # Determine finish reason
            finish_reason = "stop"
            if completion_tokens >= config.max_new_tokens:
                finish_reason = "length"

            # Record metrics
            duration_ms = (time.perf_counter() - start_time) * 1000
            metrics.increment("llm_calls")
            metrics.observe("llm_latency_ms", duration_ms)

            return GenerationResult(
                text=generated_text.strip(),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                finish_reason=finish_reason,
                model_name=self.model_name,
            )

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            # Still record the failed call for observability
            duration_ms = (time.perf_counter() - start_time) * 1000
            metrics.increment("llm_calls")
            metrics.increment("llm_errors")
            metrics.observe("llm_latency_ms", duration_ms)
            raise InferenceError(
                "Generation failed",
                details={"error": str(e), "model": self.model_name},
            ) from e

    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        system_prompt: str | None = None,
    ) -> Iterator[str]:
        """
        Stream generated text token by token.

        Args:
            prompt: Input prompt text
            config: Generation configuration
            system_prompt: Optional system prompt

        Yields:
            Generated text chunks

        Raises:
            InferenceError: If generation fails
        """
        messages = []
        if system_prompt:
            messages.append(ConversationMessage(Role.SYSTEM, system_prompt))
        messages.append(ConversationMessage(Role.USER, prompt))

        yield from self.chat_stream(messages, config)

    def chat_stream(
        self,
        messages: list[ConversationMessage],
        config: GenerationConfig | None = None,
    ) -> Iterator[str]:
        """
        Stream a chat response token by token.

        Args:
            messages: Conversation history
            config: Generation configuration

        Yields:
            Generated text chunks

        Raises:
            InferenceError: If generation fails
        """
        self.load()
        config = config or self.default_config

        try:
            # Format messages
            message_dicts = [m.to_dict() for m in messages]
            prompt_text = self._tokenizer.apply_chat_template(
                message_dicts,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Tokenize
            inputs = self._tokenizer(
                prompt_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Create streamer
            streamer = TextIteratorStreamer(
                self._tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            # Build stopping criteria
            stopping_criteria = None
            if config.stop_sequences:
                stopping_criteria = StoppingCriteriaList([
                    StopOnSequences(self._tokenizer, config.stop_sequences)
                ])

            # Start generation in background thread
            generation_kwargs = {
                **inputs,
                **config.to_generate_kwargs(),
                "streamer": streamer,
                "stopping_criteria": stopping_criteria,
                "pad_token_id": self._tokenizer.pad_token_id,
                "eos_token_id": self._tokenizer.eos_token_id,
            }

            thread = threading.Thread(
                target=self._generate_thread,
                args=(generation_kwargs,),
            )
            thread.start()

            # Yield tokens as they're generated
            buffer = ""
            for text in streamer:
                buffer += text

                # Check for stop sequences
                should_stop = False
                for seq in config.stop_sequences:
                    if seq in buffer:
                        # Yield up to stop sequence
                        idx = buffer.index(seq)
                        if idx > 0:
                            yield buffer[:idx]
                        should_stop = True
                        break

                if should_stop:
                    break

                yield text

            thread.join()

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise InferenceError(
                "Streaming generation failed",
                details={"error": str(e), "model": self.model_name},
            ) from e

    def _generate_thread(self, kwargs: dict) -> None:
        """Background thread for streaming generation."""
        with torch.no_grad():
            self._model.generate(**kwargs)

    def get_context_length(self) -> int:
        """Get the model's maximum context length."""
        self.load()

        if hasattr(self._model.config, "max_position_embeddings"):
            return self._model.config.max_position_embeddings
        elif hasattr(self._model.config, "n_positions"):
            return self._model.config.n_positions
        else:
            return 4096  # Conservative default

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to tokenize

        Returns:
            Number of tokens
        """
        self.load()
        return len(self._tokenizer.encode(text))

    def __enter__(self) -> "LocalLLM":
        """Context manager entry."""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        # Don't unload by default - let caller decide
        pass

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"LocalLLM(model={self.model_name!r}, device={self.device!r}, {status})"
