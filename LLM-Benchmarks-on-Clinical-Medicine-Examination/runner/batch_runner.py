import asyncio
from typing import Dict, List
from exam.question import Question, TestResult
from models.base import BaseLLMAdapter
from .test_runner import TestRunner


class BatchRunner:
    def __init__(self, adapters: Dict[str, BaseLLMAdapter], max_concurrent: int = 5,
                 agentic_max_rounds: int = 2,
                 json_repair_adapter: BaseLLMAdapter | None = None):
        self.adapters = adapters
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.agentic_max_rounds = agentic_max_rounds
        self.json_repair_adapter = json_repair_adapter

    async def _run_with_semaphore(self, runner: TestRunner, question: Question, mode: str) -> TestResult:
        async with self.semaphore:
            return await runner.run_single(question, mode=mode)

    async def run_all_models(
        self, questions: List[Question], mode: str = 'baseline', progress_callback=None
    ) -> Dict[str, List[TestResult]]:
        all_results = {model_id: [] for model_id in self.adapters}
        total_tasks = len(self.adapters) * len(questions)
        completed = 0

        async def run_one(model_id: str, runner: TestRunner, question: Question):
            try:
                result = await self._run_with_semaphore(runner, question, mode=mode)
                return model_id, question.id, result, None
            except Exception as e:
                return model_id, question.id, None, e

        tasks: List[asyncio.Task] = []

        for model_id, adapter in self.adapters.items():
            runner = TestRunner(
                adapter,
                model_id,
                agentic_max_rounds=self.agentic_max_rounds,
                json_repair_adapter=self.json_repair_adapter,
            )
            for question in questions:
                tasks.append(asyncio.create_task(run_one(model_id, runner, question)))

        for fut in asyncio.as_completed(tasks):
            model_id, q_id, result, err = await fut
            if err is None and result is not None:
                all_results[model_id].append(result)
            else:
                all_results[model_id].append(
                    TestResult(
                        question_id=q_id,
                        model_id=model_id,
                        mode=mode,
                        predicted_answer='',
                        correct_answer='',
                        is_correct=False,
                        score=0.0,
                        raw_response='',
                        error=str(err) if err else "Unknown error",
                        meta=None,
                    )
                )

            completed += 1
            if progress_callback:
                progress_callback(completed, total_tasks, model_id, q_id)

        for model_id in all_results:
            all_results[model_id].sort(key=lambda r: r.question_id)

        return all_results
