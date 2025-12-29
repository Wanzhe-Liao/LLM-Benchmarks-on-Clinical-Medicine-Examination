import asyncio
from typing import List
from exam.question import Question, TestResult
from exam.prompt import PromptBuilder
from exam.scorer import extract_answer, score_answer
from models.base import BaseLLMAdapter
from .agentic import AgenticSolver


class TestRunner:
    def __init__(
        self,
        adapter: BaseLLMAdapter,
        model_id: str,
        agentic_max_rounds: int = 2,
        json_repair_adapter: BaseLLMAdapter | None = None,
    ):
        self.adapter = adapter
        self.model_id = model_id
        self.agentic_solver = AgenticSolver(
            adapter,
            max_rounds=agentic_max_rounds,
            json_repair_adapter=json_repair_adapter,
        )

    async def run_single(self, question: Question, mode: str = 'baseline') -> TestResult:
        if mode not in ('baseline', 'agentic'):
            raise ValueError(f"Unknown mode: {mode}")

        if mode == 'baseline':
            prompt = PromptBuilder.build_prompt(question)
            response = await self.adapter.complete(prompt)

            predicted = ''
            if response.success:
                predicted = extract_answer(response.content)

            is_correct, score = score_answer(predicted, question.answer)

            return TestResult(
                question_id=question.id,
                model_id=self.model_id,
                mode='baseline',
                predicted_answer=predicted,
                correct_answer=question.answer,
                is_correct=is_correct,
                score=score,
                raw_response=response.content,
                latency_ms=response.latency_ms,
                error=response.error,
                meta=None
            )

        agentic = await self.agentic_solver.solve(question)
        predicted = extract_answer(agentic.final_answer)
        is_correct, score = score_answer(predicted, question.answer)

        parts = [
            "PLANNER\n" + (agentic.planner.response or ""),
            "REASONER\n" + (agentic.reasoner.response or ""),
        ]
        if agentic.critic:
            parts.append("CRITIC\n" + (agentic.critic.response or ""))
        raw = "\n\n".join(parts)

        meta = {
            "rounds": agentic.rounds,
            "latency_ms": agentic.latency_ms,
            "planner": {
                "parsed": agentic.planner.parsed,
                "latency_ms": agentic.planner.latency_ms,
                "error": agentic.planner.error,
                "repaired": agentic.planner.repaired,
                "repair_model": agentic.planner.repair_model,
                "repair_error": agentic.planner.repair_error,
            },
            "reasoner": {
                "parsed": agentic.reasoner.parsed,
                "latency_ms": agentic.reasoner.latency_ms,
                "error": agentic.reasoner.error,
                "repaired": agentic.reasoner.repaired,
                "repair_model": agentic.reasoner.repair_model,
                "repair_error": agentic.reasoner.repair_error,
            },
            "critic": None
            if not agentic.critic
            else {
                "parsed": agentic.critic.parsed,
                "latency_ms": agentic.critic.latency_ms,
                "error": agentic.critic.error,
                "repaired": agentic.critic.repaired,
                "repair_model": agentic.critic.repair_model,
                "repair_error": agentic.critic.repair_error,
            },
        }

        return TestResult(
            question_id=question.id,
            model_id=self.model_id,
            mode='agentic',
            predicted_answer=predicted,
            correct_answer=question.answer,
            is_correct=is_correct,
            score=score,
            raw_response=raw,
            latency_ms=agentic.latency_ms,
            error=agentic.error,
            meta=meta
        )

    async def run_batch(self, questions: List[Question], mode: str = 'baseline') -> List[TestResult]:
        results = []
        for question in questions:
            result = await self.run_single(question, mode=mode)
            results.append(result)

        return results
