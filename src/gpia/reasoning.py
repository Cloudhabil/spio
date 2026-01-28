"""
GPIA Reasoning Engine with LLM Integration

Connects to LLM backends (Ollama, OpenAI) for intelligent reasoning.
Implements dimension-based routing and multi-model orchestration.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, AsyncIterator
from enum import Enum
from pathlib import Path

import httpx

# Import from parent
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from sovereign_pio.constants import PHI, DIMENSION_NAMES, DIMENSION_SILICON


class ModelProvider(Enum):
    """Supported model providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class ModelConfig:
    """Configuration for a model."""
    provider: ModelProvider
    model: str
    host: str = ""
    api_key: str = ""
    temperature: float = 0.7
    max_tokens: int = 2048

    @classmethod
    def ollama(cls, model: str = "llama3.2", host: str = "http://localhost:11434"):
        return cls(provider=ModelProvider.OLLAMA, model=model, host=host)

    @classmethod
    def openai(cls, model: str = "gpt-4o-mini", api_key: str = ""):
        import os
        return cls(
            provider=ModelProvider.OPENAI,
            model=model,
            api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
        )


@dataclass
class ReasoningResult:
    """Result from a reasoning operation."""
    query: str
    response: str
    dimension: int
    dimension_name: str
    silicon: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    duration_ms: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMClient:
    """
    Unified LLM client supporting multiple providers.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self._client = httpx.AsyncClient(timeout=120.0)

    async def close(self):
        await self._client.aclose()

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate a response from the LLM."""

        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        if self.config.provider == ModelProvider.OLLAMA:
            return await self._generate_ollama(prompt, system, temp, tokens)
        elif self.config.provider == ModelProvider.OPENAI:
            return await self._generate_openai(prompt, system, temp, tokens)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

    async def _generate_ollama(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """Generate using Ollama."""

        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if system:
            payload["system"] = system

        start = time.time()
        response = await self._client.post(
            f"{self.config.host}/api/generate",
            json=payload,
        )
        response.raise_for_status()
        duration = (time.time() - start) * 1000

        data = response.json()

        return {
            "response": data.get("response", ""),
            "tokens_in": data.get("prompt_eval_count", 0),
            "tokens_out": data.get("eval_count", 0),
            "duration_ms": duration,
            "model": self.config.model,
        }

    async def _generate_openai(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """Generate using OpenAI."""

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        start = time.time()
        response = await self._client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            json={
                "model": self.config.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        response.raise_for_status()
        duration = (time.time() - start) * 1000

        data = response.json()

        return {
            "response": data["choices"][0]["message"]["content"],
            "tokens_in": data["usage"]["prompt_tokens"],
            "tokens_out": data["usage"]["completion_tokens"],
            "duration_ms": duration,
            "model": self.config.model,
        }

    async def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream response tokens."""

        if self.config.provider == ModelProvider.OLLAMA:
            async for token in self._stream_ollama(prompt, system):
                yield token
        elif self.config.provider == ModelProvider.OPENAI:
            async for token in self._stream_openai(prompt, system):
                yield token

    async def _stream_ollama(
        self,
        prompt: str,
        system: Optional[str],
    ) -> AsyncIterator[str]:
        """Stream from Ollama."""

        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": True,
        }
        if system:
            payload["system"] = system

        async with self._client.stream(
            "POST",
            f"{self.config.host}/api/generate",
            json=payload,
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]

    async def _stream_openai(
        self,
        prompt: str,
        system: Optional[str],
    ) -> AsyncIterator[str]:
        """Stream from OpenAI."""

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async with self._client.stream(
            "POST",
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            json={
                "model": self.config.model,
                "messages": messages,
                "stream": True,
            },
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    data = json.loads(data_str)
                    delta = data["choices"][0].get("delta", {})
                    if "content" in delta:
                        yield delta["content"]


class ReasoningEngine:
    """
    GPIA Reasoning Engine.

    Orchestrates reasoning across dimensions with LLM integration.
    Routes tasks to appropriate silicon (NPU/CPU/GPU) based on dimension.
    """

    # Dimension to task type mapping
    DIMENSION_TASKS = {
        1: ["perception", "sensing", "input"],
        2: ["attention", "focus", "filter"],
        3: ["security", "validation", "safety"],
        4: ["stability", "balance", "consistency"],
        5: ["compression", "encoding", "summary"],
        6: ["harmony", "coordination", "sync"],
        7: ["reasoning", "logic", "analysis"],
        8: ["prediction", "forecast", "anticipation"],
        9: ["creativity", "generation", "synthesis"],
        10: ["wisdom", "judgment", "decision"],
        11: ["integration", "combination", "merge"],
        12: ["unification", "transcendence", "holistic"],
    }

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig.ollama()
        self.client = LLMClient(self.config)
        self.phi = PHI

    async def close(self):
        """Close the LLM client."""
        await self.client.close()

    def route_to_dimension(self, task_type: str) -> int:
        """Route a task type to its appropriate dimension."""
        task_lower = task_type.lower()

        for dim, tasks in self.DIMENSION_TASKS.items():
            if any(t in task_lower for t in tasks):
                return dim

        # Default to reasoning (dimension 7)
        return 7

    def get_system_prompt(self, dimension: int) -> str:
        """Get dimension-specific system prompt."""

        prompts = {
            1: "You are a perception agent. Focus on accurately observing and describing inputs.",
            2: "You are an attention agent. Filter and prioritize the most relevant information.",
            3: "You are a security agent. Validate inputs and ensure safety constraints.",
            4: "You are a stability agent. Maintain consistency and balance in responses.",
            5: "You are a compression agent. Summarize and encode information efficiently.",
            6: "You are a harmony agent. Coordinate and synchronize multiple elements.",
            7: "You are a reasoning agent. Apply logic and analysis to solve problems.",
            8: "You are a prediction agent. Forecast outcomes based on patterns.",
            9: "You are a creativity agent. Generate novel ideas and solutions.",
            10: "You are a wisdom agent. Apply judgment and make decisions.",
            11: "You are an integration agent. Combine multiple perspectives.",
            12: "You are a unification agent. Synthesize holistic understanding.",
        }

        base = prompts.get(dimension, prompts[7])
        return f"{base}\n\nOperate with PHI-based determinism. Be precise and consistent."

    async def reason(
        self,
        query: str,
        task_type: str = "reasoning",
        context: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> ReasoningResult:
        """
        Perform reasoning on a query.

        Args:
            query: The query to reason about
            task_type: Type of task for dimension routing
            context: Optional context to include
            temperature: Override temperature

        Returns:
            ReasoningResult with response and metadata
        """
        # Route to dimension
        dimension = self.route_to_dimension(task_type)
        dimension_name = DIMENSION_NAMES[dimension]
        silicon = DIMENSION_SILICON[dimension]

        # Build prompt
        prompt = query
        if context:
            prompt = f"Context:\n{context}\n\nQuery:\n{query}"

        # Get system prompt for dimension
        system = self.get_system_prompt(dimension)

        # Generate response
        result = await self.client.generate(
            prompt=prompt,
            system=system,
            temperature=temperature,
        )

        return ReasoningResult(
            query=query,
            response=result["response"],
            dimension=dimension,
            dimension_name=dimension_name,
            silicon=silicon,
            model=result["model"],
            tokens_in=result["tokens_in"],
            tokens_out=result["tokens_out"],
            duration_ms=result["duration_ms"],
            metadata={"task_type": task_type, "context_provided": context is not None},
        )

    async def reason_stream(
        self,
        query: str,
        task_type: str = "reasoning",
    ) -> AsyncIterator[str]:
        """Stream reasoning response."""
        dimension = self.route_to_dimension(task_type)
        system = self.get_system_prompt(dimension)

        async for token in self.client.stream(query, system):
            yield token


class MultiModelOrchestrator:
    """
    Orchestrates multiple models across dimensions.

    Assigns specialized models to different dimensions for optimal performance.
    """

    def __init__(self):
        self.models: Dict[int, ModelConfig] = {}
        self.engines: Dict[int, ReasoningEngine] = {}
        self._default_config = ModelConfig.ollama()

    def set_model(self, dimension: int, config: ModelConfig):
        """Set model for a specific dimension."""
        self.models[dimension] = config
        self.engines[dimension] = ReasoningEngine(config)

    def set_default(self, config: ModelConfig):
        """Set default model for unmapped dimensions."""
        self._default_config = config

    def get_engine(self, dimension: int) -> ReasoningEngine:
        """Get engine for dimension, creating if needed."""
        if dimension not in self.engines:
            config = self.models.get(dimension, self._default_config)
            self.engines[dimension] = ReasoningEngine(config)
        return self.engines[dimension]

    async def reason(
        self,
        query: str,
        task_type: str = "reasoning",
        **kwargs,
    ) -> ReasoningResult:
        """Route query to appropriate model based on dimension."""
        # Determine dimension
        temp_engine = ReasoningEngine(self._default_config)
        dimension = temp_engine.route_to_dimension(task_type)

        # Get engine for dimension
        engine = self.get_engine(dimension)

        return await engine.reason(query, task_type, **kwargs)

    async def close(self):
        """Close all engines."""
        for engine in self.engines.values():
            await engine.close()
