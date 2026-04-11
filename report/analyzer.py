from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from exam.question import Question, TestResult
from exam.scorer import calculate_accuracy, calculate_accuracy_by_type


@dataclass
class ModelStats:
    model_id: str
    total_correct: int
    total_count: int
    accuracy: float
    by_type: Dict[str, Dict]


@dataclass
class ComparisonStats:
    model_id: str
    baseline_accuracy: float
    agentic_accuracy: float
    improvement: float
    changed_answers: int
    improved_answers: int
    degraded_answers: int


@dataclass
class ConsistencyStats:
    all_correct: List[int]
    all_wrong: List[int]
    most_divisive: List[Tuple[int, int]]


class ResultAnalyzer:
    def __init__(self, questions: List[Question]):
        self.questions = questions
        self.q_map = {q.id: q for q in questions}

    def _get_question_id(self, r) -> int:
        if hasattr(r, "question_id"):
            return int(getattr(r, "question_id") or 0)
        if isinstance(r, dict):
            return int(r.get("question_id", 0) or 0)
        return 0

    def _is_correct(self, r) -> bool:
        if hasattr(r, "is_correct"):
            return bool(getattr(r, "is_correct"))
        if isinstance(r, dict):
            return bool(r.get("is_correct", False))
        return False

    def _get_predicted(self, r) -> str:
        if hasattr(r, "predicted_answer"):
            return str(getattr(r, "predicted_answer") or "")
        if isinstance(r, dict):
            return str(r.get("predicted_answer", "") or "")
        return ""

    def _get_model_id(self, r) -> str:
        if hasattr(r, "model_id"):
            return str(getattr(r, "model_id") or "")
        if isinstance(r, dict):
            return str(r.get("model_id", "") or "")
        return ""

    def analyze_model(self, results: List[TestResult]) -> ModelStats:
        accuracy_stats = calculate_accuracy(results)
        by_type_stats = calculate_accuracy_by_type(results, self.questions)

        return ModelStats(
            model_id=self._get_model_id(results[0]) if results else '',
            total_correct=accuracy_stats['correct'],
            total_count=accuracy_stats['total_count'],
            accuracy=accuracy_stats['total'],
            by_type=by_type_stats
        )

    def compare_modes(self, baseline_results: List[TestResult], agentic_results: List[TestResult]) -> ComparisonStats:
        model_id = self._get_model_id(baseline_results[0]) if baseline_results else (self._get_model_id(agentic_results[0]) if agentic_results else '')

        baseline_map = {self._get_question_id(r): r for r in baseline_results}
        agentic_map = {self._get_question_id(r): r for r in agentic_results}

        changed = 0
        improved = 0
        degraded = 0

        for q_id, base in baseline_map.items():
            ag = agentic_map.get(q_id)
            if not ag:
                continue

            if self._get_predicted(base) != self._get_predicted(ag):
                changed += 1
                if not self._is_correct(base) and self._is_correct(ag):
                    improved += 1
                elif self._is_correct(base) and not self._is_correct(ag):
                    degraded += 1

        baseline_acc = calculate_accuracy(baseline_results)['total']
        agentic_acc = calculate_accuracy(agentic_results)['total']

        return ComparisonStats(
            model_id=model_id,
            baseline_accuracy=baseline_acc,
            agentic_accuracy=agentic_acc,
            improvement=agentic_acc - baseline_acc,
            changed_answers=changed,
            improved_answers=improved,
            degraded_answers=degraded
        )

    def analyze_consistency(self, all_results: Dict[str, List[TestResult]]) -> ConsistencyStats:
        question_ids = set()
        for results in all_results.values():
            for r in results:
                question_ids.add(self._get_question_id(r))

        all_correct = []
        all_wrong = []
        divisive = []

        for q_id in sorted(question_ids):
            correct_count = 0
            total_count = 0

            for model_id, results in all_results.items():
                for r in results:
                    if self._get_question_id(r) == q_id:
                        total_count += 1
                        if self._is_correct(r):
                            correct_count += 1
                        break

            if total_count > 0:
                if correct_count == total_count:
                    all_correct.append(q_id)
                elif correct_count == 0:
                    all_wrong.append(q_id)
                else:
                    divisive.append((q_id, correct_count))

        divisive.sort(key=lambda x: abs(x[1] - len(all_results) / 2))

        return ConsistencyStats(
            all_correct=all_correct,
            all_wrong=all_wrong,
            most_divisive=divisive[:10]
        )

    def get_wrong_questions_detail(self, all_results: Dict[str, List[TestResult]],
                                   wrong_ids: List[int]) -> List[Dict]:
        details = []

        for q_id in wrong_ids:
            question = self.q_map.get(q_id)
            if not question:
                continue

            model_answers = {}
            for model_id, results in all_results.items():
                for r in results:
                    if self._get_question_id(r) == q_id:
                        model_answers[model_id] = self._get_predicted(r)
                        break

            details.append({
                'id': q_id,
                'type': question.type,
                'stem': question.stem[:100] + '...' if len(question.stem) > 100 else question.stem,
                'correct_answer': question.answer,
                'model_answers': model_answers
            })

        return details
