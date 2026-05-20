from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

DEFAULT_MODEL = "Qwen/Qwen3-4B"
DEFAULT_PROMPTS = Path("data/prompts.jsonl")
DEFAULT_OUTPUT = Path("results/baseline.json")


@dataclass(frozen=True)
class PromptRecord:
    id: str
    prompt: str


@dataclass(frozen=True)
class BenchmarkResult:
    model: str
    prompts_path: str
    num_requests: int
    max_new_tokens: int
    batch_size: int
    tensor_parallel_size: int
    gpu_memory_utilization: float
    total_runtime_seconds: float
    total_generated_tokens: int
    tokens_per_second: float
    requests_per_second: float
    average_latency_seconds: float
    p95_latency_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a baseline vLLM inference benchmark."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_prompts(path: Path) -> list[PromptRecord]:
    records: list[PromptRecord] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

            prompt = payload.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    f"Line {line_number} must contain a non-empty string field: prompt"
                )

            prompt_id = payload.get("id", f"prompt-{line_number}")
            if not isinstance(prompt_id, str) or not prompt_id.strip():
                prompt_id = f"prompt-{line_number}"

            records.append(PromptRecord(id=prompt_id, prompt=prompt))

    if not records:
        raise ValueError(f"No prompts found in {path}")

    return records


def batched(items: list[PromptRecord], batch_size: int) -> Iterable[list[PromptRecord]]:
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def count_generated_tokens(output: Any) -> int:
    completions = getattr(output, "outputs", [])
    if not completions:
        return 0

    token_ids = getattr(completions[0], "token_ids", None)
    if token_ids is not None:
        return len(token_ids)

    text = getattr(completions[0], "text", "")
    return len(text.split())


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]

    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * percentile_value
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = rank - lower_index
    return (
        sorted_values[lower_index] * (1.0 - weight)
        + sorted_values[upper_index] * weight
    )


def run_benchmark(
    model: str,
    prompts_path: Path,
    max_new_tokens: int,
    batch_size: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
) -> BenchmarkResult:
    from vllm import LLM, SamplingParams

    prompts = load_prompts(prompts_path)
    if max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1")

    sampling_params = SamplingParams(max_tokens=max_new_tokens)
    llm = LLM(
        model=model,
        trust_remote_code=True,
        dtype="auto",
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    total_generated_tokens = 0
    request_latencies: list[float] = []

    benchmark_start = time.perf_counter()
    for batch in batched(prompts, batch_size):
        batch_prompts = [record.prompt for record in batch]

        batch_start = time.perf_counter()
        outputs = llm.generate(batch_prompts, sampling_params)
        batch_elapsed = time.perf_counter() - batch_start

        request_latencies.extend([batch_elapsed] * len(batch))
        total_generated_tokens += sum(
            count_generated_tokens(output) for output in outputs
        )

    total_runtime = time.perf_counter() - benchmark_start
    num_requests = len(prompts)

    return BenchmarkResult(
        model=model,
        prompts_path=str(prompts_path),
        num_requests=num_requests,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        total_runtime_seconds=total_runtime,
        total_generated_tokens=total_generated_tokens,
        tokens_per_second=(
            total_generated_tokens / total_runtime if total_runtime > 0 else 0.0
        ),
        requests_per_second=num_requests / total_runtime if total_runtime > 0 else 0.0,
        average_latency_seconds=statistics.fmean(request_latencies)
        if request_latencies
        else 0.0,
        p95_latency_seconds=percentile(request_latencies, 0.95),
    )


def save_result(result: BenchmarkResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(asdict(result), file, indent=2)
        file.write("\n")


def print_summary(result: BenchmarkResult, output_path: Path) -> None:
    print("\nBaseline Benchmark Summary")
    print("==========================")
    print(f"Model:                  {result.model}")
    print(f"Prompts:                {result.num_requests}")
    print(f"Max new tokens:         {result.max_new_tokens}")
    print(f"Batch size:             {result.batch_size}")
    print(f"Tensor parallel size:   {result.tensor_parallel_size}")
    print(f"GPU memory utilization: {result.gpu_memory_utilization:.2f}")
    print(f"Total runtime:          {result.total_runtime_seconds:.2f}s")
    print(f"Generated tokens:       {result.total_generated_tokens}")
    print(f"Tokens/sec:             {result.tokens_per_second:.2f}")
    print(f"Requests/sec:           {result.requests_per_second:.2f}")
    print(f"Average latency:        {result.average_latency_seconds:.2f}s")
    print(f"P95 latency:            {result.p95_latency_seconds:.2f}s")
    print(f"Saved result:           {output_path}")


def main() -> None:
    args = parse_args()
    result = run_benchmark(
        model=args.model,
        prompts_path=args.prompts,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    save_result(result, args.output)
    print_summary(result, args.output)


if __name__ == "__main__":
    main()
