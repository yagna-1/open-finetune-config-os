#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import types
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import nbformat as nbf
from nbformat import validate as validate_notebook

from ft_config_engine.models import RecommendationRequest
from ft_config_engine.recommender import build_engine_from_dataset


@dataclass(frozen=True)
class SmokeScenario:
    name: str
    expected_template: str
    request: RecommendationRequest


@dataclass(frozen=True)
class SmokeResult:
    name: str
    template: str
    selected_gpu: str
    notebook_cells: int
    code_cells: int
    compile_ok: bool
    execution_ok: bool


SCENARIOS: tuple[SmokeScenario, ...] = (
    SmokeScenario(
        name="classification_small_colab",
        expected_template="classification",
        request=RecommendationRequest(
            platform="colab",
            plan="free",
            task_type="classification",
            adapter_type="none",
            model_size_bucket="small",
            model_name="bert-base-uncased",
            dataset_name="imdb",
            strategy="deterministic",
        ),
    ),
    SmokeScenario(
        name="qa_small_kaggle",
        expected_template="qa",
        request=RecommendationRequest(
            platform="kaggle",
            plan="free",
            task_type="question_answering",
            adapter_type="none",
            model_size_bucket="small",
            model_name="bert-base-uncased",
            dataset_name="squad",
            strategy="deterministic",
        ),
    ),
    SmokeScenario(
        name="summarization_small_colab",
        expected_template="summarization_t5",
        request=RecommendationRequest(
            platform="colab",
            plan="pro",
            task_type="summarization",
            adapter_type="none",
            model_size_bucket="small",
            model_name="t5-small",
            dataset_name="cnn_dailymail",
            strategy="deterministic",
        ),
    ),
    SmokeScenario(
        name="causal_lora_lightning",
        expected_template="causal_lm_lora",
        request=RecommendationRequest(
            platform="lightning",
            plan="pro",
            task_type="code_generation",
            adapter_type="lora",
            model_size_bucket="medium",
            model_name="bigcode/starcoderbase-1b",
            dataset_name="code_alpaca",
            strategy="deterministic",
        ),
    ),
    SmokeScenario(
        name="qlora_chat_colab",
        expected_template="qlora_4bit",
        request=RecommendationRequest(
            platform="colab",
            plan="free",
            task_type="chatbot",
            adapter_type="qlora",
            model_size_bucket="medium",
            model_name="meta-llama/Llama-2-7b-hf",
            dataset_name="alpaca",
            strategy="deterministic",
        ),
    ),
)


class FakeSplit:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.column_names = list(rows[0].keys()) if rows else []

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._rows[index]

    def batch(self) -> dict[str, list[Any]]:
        keys = list(self._rows[0].keys()) if self._rows else []
        return {key: [row.get(key) for row in self._rows] for key in keys}


class FakeDatasetDict:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._splits = {
            "train": FakeSplit(rows),
            "validation": FakeSplit(rows[:1] or rows),
            "test": FakeSplit(rows[:1] or rows),
        }

    def __contains__(self, key: str) -> bool:
        return key in self._splits

    def __iter__(self):
        return iter(self._splits)

    def keys(self):
        return self._splits.keys()

    def __getitem__(self, key: str) -> FakeSplit:
        return self._splits[key]

    def map(
        self,
        fn,
        *,
        batched: bool = False,
        remove_columns: Iterable[str] | None = None,
        **_: Any,
    ) -> "FakeDatasetDict":
        del remove_columns
        train_split = self._splits["train"]
        if batched:
            fn(train_split.batch())
        else:
            for row in train_split._rows:
                fn(row)
        return self


class FakeTokenizer:
    pad_token: str | None = None
    eos_token: str = "<eos>"

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs) -> "FakeTokenizer":
        return cls()

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if "text_target" in kwargs and isinstance(kwargs["text_target"], list):
            batch_size = len(kwargs["text_target"])
        elif args and isinstance(args[0], list):
            batch_size = len(args[0])
        else:
            batch_size = 1
        return {
            "input_ids": [[1, 2, 3] for _ in range(batch_size)],
            "attention_mask": [[1, 1, 1] for _ in range(batch_size)],
        }

    def save_pretrained(self, _path: str) -> None:
        return None

    def push_to_hub(self, _repo_id: str) -> None:
        return None


class FakeModel:
    def __init__(self) -> None:
        self.config = types.SimpleNamespace(use_cache=True)

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs) -> "FakeModel":
        return cls()

    def save_pretrained(self, _path: str) -> None:
        return None

    def push_to_hub(self, _repo_id: str) -> None:
        return None

    def gradient_checkpointing_enable(self) -> None:
        return None


class FakeArgs:
    def __init__(self, **_kwargs: Any) -> None:
        return None


class FakeTrainer:
    def __init__(self, **kwargs: Any) -> None:
        self._tokenizer = kwargs.get("tokenizer")

    def train(self) -> dict[str, str]:
        return {"status": "ok"}

    def save_model(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)

    def push_to_hub(self, _repo_id: str) -> None:
        return None


class FakeCollator:
    def __init__(self, **_kwargs: Any) -> None:
        return None


class FakeBitsAndBytesConfig:
    def __init__(self, **_kwargs: Any) -> None:
        return None


class FakeLoraConfig:
    def __init__(self, **_kwargs: Any) -> None:
        return None


class FakeTaskType:
    CAUSAL_LM = "CAUSAL_LM"
    SEQ_CLS = "SEQ_CLS"
    SEQ_2_SEQ_LM = "SEQ_2_SEQ_LM"
    QUESTION_ANS = "QUESTION_ANS"


@contextmanager
def patched_notebook_runtime() -> Any:
    original_modules = {}
    module_names = ["torch", "datasets", "transformers", "peft", "huggingface_hub"]
    for name in module_names:
        original_modules[name] = sys.modules.get(name)

    torch_mod = types.ModuleType("torch")

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def get_device_name(_index: int) -> str:
            return "FakeGPU"

    torch_mod.cuda = _Cuda()
    torch_mod.float16 = "float16"
    torch_mod.bfloat16 = "bfloat16"
    torch_mod.float32 = "float32"
    sys.modules["torch"] = torch_mod

    datasets_mod = types.ModuleType("datasets")

    def _load_dataset(_name: str) -> FakeDatasetDict:
        rows = [
            {
                "text": "sample text",
                "label": 1,
                "labels": [1, 2, 3],
                "question": "What is this?",
                "context": "This is a sample context.",
                "document": "This is a sample document.",
                "summary": "sample summary",
                "instruction": "Do a task",
                "prompt": "Explain this",
                "input": "input value",
                "content": "content value",
            },
            {
                "text": "another sample",
                "label": 0,
                "labels": [1, 2, 3],
                "question": "How does this work?",
                "context": "Another sample context.",
                "document": "Another sample document.",
                "summary": "another summary",
                "instruction": "Do another task",
                "prompt": "Summarize this",
                "input": "another input",
                "content": "another content",
            },
        ]
        return FakeDatasetDict(rows=rows)

    datasets_mod.load_dataset = _load_dataset
    sys.modules["datasets"] = datasets_mod

    transformers_mod = types.ModuleType("transformers")
    transformers_mod.AutoTokenizer = FakeTokenizer
    transformers_mod.AutoModelForCausalLM = FakeModel
    transformers_mod.AutoModelForSequenceClassification = FakeModel
    transformers_mod.AutoModelForSeq2SeqLM = FakeModel
    transformers_mod.AutoModelForQuestionAnswering = FakeModel
    transformers_mod.BitsAndBytesConfig = FakeBitsAndBytesConfig
    transformers_mod.Trainer = FakeTrainer
    transformers_mod.Seq2SeqTrainer = FakeTrainer
    transformers_mod.TrainingArguments = FakeArgs
    transformers_mod.Seq2SeqTrainingArguments = FakeArgs
    transformers_mod.DataCollatorForLanguageModeling = FakeCollator
    transformers_mod.DataCollatorWithPadding = FakeCollator
    transformers_mod.DataCollatorForSeq2Seq = FakeCollator
    transformers_mod.default_data_collator = object()
    sys.modules["transformers"] = transformers_mod

    peft_mod = types.ModuleType("peft")
    peft_mod.LoraConfig = FakeLoraConfig
    peft_mod.TaskType = FakeTaskType
    peft_mod.get_peft_model = lambda model, _config: model
    peft_mod.prepare_model_for_kbit_training = lambda model: model
    sys.modules["peft"] = peft_mod

    hf_hub_mod = types.ModuleType("huggingface_hub")
    hf_hub_mod.login = lambda **_kwargs: None
    sys.modules["huggingface_hub"] = hf_hub_mod

    import subprocess

    original_check_call = subprocess.check_call
    subprocess.check_call = lambda *_args, **_kwargs: 0
    try:
        yield
    finally:
        subprocess.check_call = original_check_call
        for name, module in original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@contextmanager
def in_workdir(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated notebook templates end-to-end")
    parser.add_argument(
        "--dataset",
        default="finetuning_configs_final.jsonl",
        help="Dataset path(s); accepts comma-separated values",
    )
    parser.add_argument("--ml-reranker", dest="ml_reranker_path")
    parser.add_argument("--hp-predictor", dest="hp_predictor_path")
    parser.add_argument(
        "--out-dir",
        default="artifacts/notebook_smoke",
        help="Directory where generated notebook artifacts and report are written",
    )
    return parser.parse_args()


def validate_structure(notebook_json: dict[str, Any], scenario_name: str) -> tuple[int, int]:
    notebook = nbf.from_dict(notebook_json)
    validate_notebook(notebook)

    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    markdown_cells = [cell for cell in notebook.cells if cell.cell_type == "markdown"]
    if not code_cells:
        raise AssertionError(f"{scenario_name}: notebook has no code cells")
    if not markdown_cells:
        raise AssertionError(f"{scenario_name}: notebook has no markdown cells")

    joined_code = "\n".join(_cell_source(cell) for cell in code_cells)
    required_snippets = [
        "pip",
        "torch.cuda.is_available",
        "trainer.train()",
    ]
    for snippet in required_snippets:
        if snippet not in joined_code:
            raise AssertionError(f"{scenario_name}: missing required snippet '{snippet}'")

    return len(notebook.cells), len(code_cells)


def compile_cells(notebook_json: dict[str, Any], scenario_name: str) -> None:
    notebook = nbf.from_dict(notebook_json)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        filename = f"<{scenario_name}:cell_{index}>"
        compile(_cell_source(cell), filename, "exec")


def execute_cells(notebook_json: dict[str, Any], scenario_name: str) -> None:
    notebook = nbf.from_dict(notebook_json)
    namespace: dict[str, Any] = {
        "__name__": "__notebook__",
        "__file__": f"{scenario_name}.ipynb",
        "print": lambda *_args, **_kwargs: None,
    }
    with tempfile.TemporaryDirectory(prefix="ft_nb_smoke_") as temp_dir:
        temp_path = Path(temp_dir)
        with in_workdir(temp_path):
            with patched_notebook_runtime():
                for index, cell in enumerate(notebook.cells):
                    if cell.cell_type != "code":
                        continue
                    code = compile(_cell_source(cell), f"<{scenario_name}:run_{index}>", "exec")
                    exec(code, namespace, namespace)  # noqa: S102


def _cell_source(cell: Any) -> str:
    source = cell.source
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def run() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = build_engine_from_dataset(
        args.dataset,
        ml_reranker_path=args.ml_reranker_path,
        hp_predictor_path=args.hp_predictor_path,
    )

    results: list[SmokeResult] = []
    for scenario in SCENARIOS:
        recommendation = engine.recommend(scenario.request, render_notebook=True)
        if recommendation.notebook_json is None:
            raise AssertionError(f"{scenario.name}: notebook payload is empty")
        if recommendation.notebook_template != scenario.expected_template:
            raise AssertionError(
                f"{scenario.name}: expected template {scenario.expected_template}, "
                f"got {recommendation.notebook_template}"
            )

        notebook_path = out_dir / f"{scenario.name}.ipynb"
        notebook_path.write_text(
            json.dumps(recommendation.notebook_json, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

        notebook_cells, code_cells = validate_structure(recommendation.notebook_json, scenario.name)
        compile_cells(recommendation.notebook_json, scenario.name)
        execute_cells(recommendation.notebook_json, scenario.name)

        results.append(
            SmokeResult(
                name=scenario.name,
                template=recommendation.notebook_template,
                selected_gpu=recommendation.selected_gpu,
                notebook_cells=notebook_cells,
                code_cells=code_cells,
                compile_ok=True,
                execution_ok=True,
            )
        )

    report = {"scenarios": [asdict(row) for row in results], "count": len(results)}
    report_path = out_dir / "smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Notebook smoke validation passed for {len(results)} scenarios")
    print(f"Artifacts written to: {out_dir}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    run()
