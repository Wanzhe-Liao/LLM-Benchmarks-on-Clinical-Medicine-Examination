from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List


@dataclass
class Question:
    id: int
    type: str  # 'A', 'B', 'X'
    stem: str
    options: Dict[str, str]
    answer: str
    shared_stem: Optional[str] = None
    shared_options: Optional[Dict[str, str]] = None

    def get_full_stem(self) -> str:
        if self.shared_stem:
            return f"{self.shared_stem}\n\n{self.stem}"
        return self.stem

    def get_full_options(self) -> Dict[str, str]:
        if self.shared_options:
            return self.shared_options
        return self.options

    def format_options(self) -> str:
        opts = self.get_full_options()
        return '\n'.join([f"{k}. {v}" for k, v in sorted(opts.items())])

    def is_multiple_choice(self) -> bool:
        return self.type == 'X'


@dataclass
class TestResult:
    question_id: int
    model_id: str
    mode: str  # 'baseline' or 'agentic'
    predicted_answer: str
    correct_answer: str
    is_correct: bool
    score: float
    raw_response: str
    latency_ms: int = 0
    error: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
