import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import time

from exam.question import Question
from exam.scorer import extract_answer
from models.base import BaseLLMAdapter


def format_question_for_agents(question: Question) -> str:
    stem = question.get_full_stem().strip()
    options = question.format_options().strip()
    return f"题型: {question.type}\n\n题干:\n{stem}\n\n选项:\n{options}\n"


def _strip_fences(text: str) -> str:
    t = text or ""
    t = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", t, flags=re.IGNORECASE)
    return t.strip()


def _try_load_json_object(candidate: str) -> Optional[Dict[str, Any]]:
    if not candidate:
        return None
    c = candidate.strip()
    try:
        obj = json.loads(c)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    # remove trailing commas
    c2 = re.sub(r",\s*([}\]])", r"\1", c)
    if c2 != c:
        try:
            obj = json.loads(c2)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass

    return None


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    cleaned = _strip_fences(text)
    cleaned = cleaned.strip()

    if cleaned.startswith("{") and cleaned.endswith("}"):
        direct = _try_load_json_object(cleaned)
        if direct:
            return direct

    # Scan for the first balanced {...} outside of quotes.
    in_str = False
    quote = ""
    escape = False
    depth = 0
    start = -1

    for i, ch in enumerate(cleaned):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if in_str:
            if ch == quote:
                in_str = False
            continue

        if ch in ("\"", "'"):
            in_str = True
            quote = ch
            continue

        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    candidate = cleaned[start : i + 1]
                    obj = _try_load_json_object(candidate)
                    if obj:
                        return obj
                    start = -1

    return None


@dataclass
class AgentOutput:
    prompt: str
    response: str
    parsed: Optional[Dict[str, Any]] = None
    latency_ms: int = 0
    error: Optional[str] = None
    repaired: bool = False
    repair_model: Optional[str] = None
    repair_error: Optional[str] = None


@dataclass
class AgenticSolveResult:
    final_answer: str
    planner: AgentOutput
    reasoner: AgentOutput
    critic: Optional[AgentOutput]
    rounds: int
    error: Optional[str] = None
    latency_ms: int = 0


class PlannerAgent:
    def build_prompt(self, question_text: str) -> str:
        return (
            "你是规划智能体（Planner Agent）。你不直接回答选项。\n"
            "请对下面医学选择题进行审题与拆解，输出 JSON（仅 JSON）：\n"
            "约束：\n"
            "- 不要使用 ``` 代码块，不要输出任何解释文本；只输出一个 JSON 对象。\n"
            "- 列表字段每个最多 6 条，每条尽量简短（≤30字）。\n"
            "{\n"
            "  \"domain\": \"生理/生化|病理学|内科|外科|其他\",\n"
            "  \"key_facts\": [\"...\"],\n"
            "  \"sub_questions\": [\"...\"],\n"
            "  \"risk_checks\": [\"需要特别警惕的禁忌/红旗/陷阱...\"],\n"
            "  \"answer_type\": \"single|multiple\",\n"
            "  \"notes\": \"...\"\n"
            "}\n\n"
            f"{question_text}"
        )


class ReasoningAgent:
    def build_prompt(self, question_text: str, plan_json: Dict[str, Any], critic_feedback: str = "") -> str:
        feedback_block = f"\n\n质控反馈（如有）：\n{critic_feedback}\n" if critic_feedback.strip() else ""
        return (
            "你是推理智能体（Reasoning Agent），扮演资深的临床科研工作者。\n"
            "基于规划智能体给出的子问题与风险点，进行分步推理并选择正确选项。\n"
            "要求：\n"
            "- 不要使用 ``` 代码块；只输出一个 JSON 对象。\n"
            "- steps 必须是高层次、简洁、可审核的医学理由：最多 6 条，每条 ≤30字，避免大段推理。\n"
            "- 最终只在 JSON 的 final_answer 字段给出答案字母（如 A 或 AB），不要包含其它字符。\n"
            "- 不要输出多个候选答案；如不确定也必须给出最可能答案。\n"
            "输出 JSON（仅 JSON）：\n"
            "{\n"
            "  \"steps\": [\"1) ...\", \"2) ...\"],\n"
            "  \"final_answer\": \"A|AB|...\",\n"
            "  \"uncertainties\": [\"...\"],\n"
            "  \"confidence\": 0.0\n"
            "}\n\n"
            "规划结果 JSON：\n"
            f"{json.dumps(plan_json, ensure_ascii=False)}\n"
            f"{feedback_block}\n"
            f"{question_text}"
        )


class CriticAgent:
    def build_prompt(self, question_text: str, plan_json: Dict[str, Any], reasoning_json: Dict[str, Any]) -> str:
        return (
            "你是批评智能体（Critic Agent），扮演上级医师/质控员。\n"
            "请检查推理是否：\n"
            "- 对齐题干条件（否定词、限制条件、时间窗）\n"
            "- 覆盖关键鉴别诊断与禁忌/红旗\n"
            "- 结论是否与步骤一致，是否出现逻辑跳跃\n"
            "输出 JSON（仅 JSON）：（不要使用 ``` 代码块）\n"
            "{\n"
            "  \"verdict\": \"approve|reject\",\n"
            "  \"issues\": [\"...\"],\n"
            "  \"required_fixes\": \"...\"\n"
            "}\n\n"
            "规划结果 JSON：\n"
            f"{json.dumps(plan_json, ensure_ascii=False)}\n\n"
            "推理结果 JSON：\n"
            f"{json.dumps(reasoning_json, ensure_ascii=False)}\n\n"
            f"{question_text}"
        )


class AgenticSolver:
    """
    Planner -> Reasoner -> (Critic -> optional re-run Reasoner)
    All agents share the same underlying adapter to isolate the effect of agentic orchestration.
    """

    def __init__(self, adapter: BaseLLMAdapter, max_rounds: int = 2, json_repair_adapter: BaseLLMAdapter | None = None):
        self.adapter = adapter
        self.max_rounds = max(1, max_rounds)
        self.json_repair_adapter = json_repair_adapter
        self.planner = PlannerAgent()
        self.reasoner = ReasoningAgent()
        self.critic = CriticAgent()

    async def _parse_or_repair(
        self,
        role: str,
        raw_text: str,
        schema_example: Dict[str, Any],
    ) -> Tuple[Optional[Dict[str, Any]], bool, Optional[str]]:
        parsed = _extract_first_json_object(raw_text)
        if parsed:
            return parsed, False, None

        if not self.json_repair_adapter:
            return None, False, f"{role} json parse failed"

        prompt = (
            "你是一个严格的 JSON 修复/提取器。\n"
            "任务：把【原始输出】转换成【严格合法的 JSON 对象】。\n"
            "要求：\n"
            "- 只输出 JSON 对象，不要任何解释文字/代码块。\n"
            "- 必须包含 schema_example 中出现的所有 key，并保持类型一致。\n"
            "- 如原始输出缺失信息，给出最合理的默认值（但不要输出空对象）。\n\n"
            "schema_example:\n"
            f"{json.dumps(schema_example, ensure_ascii=False)}\n\n"
            "原始输出:\n"
            f"{raw_text}\n"
        )

        repaired = await self.json_repair_adapter.complete(prompt)
        if repaired.error:
            return None, True, repaired.error

        parsed2 = _extract_first_json_object(repaired.content) or _try_load_json_object(_strip_fences(repaired.content))
        if parsed2:
            return parsed2, True, None

        return None, True, f"{role} json repair parse failed"

    async def solve(self, question: Question) -> AgenticSolveResult:
        start_time = time.time()
        question_text = format_question_for_agents(question)

        planner_prompt = self.planner.build_prompt(question_text)
        planner_resp = await self.adapter.complete(planner_prompt)
        planner_text = planner_resp.content if planner_resp.success else ""
        planner_json, planner_repaired, planner_repair_error = await self._parse_or_repair(
            "planner",
            planner_text,
            {
                "domain": "生理/生化|病理学|内科|外科|其他",
                "key_facts": ["..."],
                "sub_questions": ["..."],
                "risk_checks": ["..."],
                "answer_type": "single|multiple",
                "notes": "...",
            },
        )
        planner_out = AgentOutput(
            prompt=planner_prompt,
            response=planner_text,
            parsed=planner_json or None,
            latency_ms=planner_resp.latency_ms,
            error=planner_resp.error,
            repaired=planner_repaired,
            repair_model=(self.json_repair_adapter.model_name if (planner_repaired and self.json_repair_adapter) else None),
            repair_error=planner_repair_error,
        )

        critic_out: Optional[AgentOutput] = None
        reasoner_out: Optional[AgentOutput] = None
        error: Optional[str] = planner_resp.error or planner_repair_error

        critic_feedback = ""
        last_candidate_answer = ""
        rounds_used = 0

        if planner_resp.error or not planner_json:
            latency_ms = int((time.time() - start_time) * 1000)
            return AgenticSolveResult(
                final_answer="",
                planner=planner_out,
                reasoner=AgentOutput(prompt="", response="", parsed=None),
                critic=None,
                rounds=0,
                error=error or "planner failed",
                latency_ms=latency_ms,
            )

        for round_idx in range(self.max_rounds):
            rounds_used = round_idx + 1
            reasoner_prompt = self.reasoner.build_prompt(question_text, planner_json, critic_feedback=critic_feedback)
            reasoner_resp = await self.adapter.complete(reasoner_prompt)
            reasoner_text = reasoner_resp.content if reasoner_resp.success else ""
            reasoning_json, reasoner_repaired, reasoner_repair_error = await self._parse_or_repair(
                "reasoner",
                reasoner_text,
                {
                    "steps": ["1) ..."],
                    "final_answer": "A|AB|...",
                    "uncertainties": ["..."],
                    "confidence": 0.0,
                },
            )
            reasoner_out = AgentOutput(
                prompt=reasoner_prompt,
                response=reasoner_text,
                parsed=reasoning_json or None,
                latency_ms=reasoner_resp.latency_ms,
                error=reasoner_resp.error,
                repaired=reasoner_repaired,
                repair_model=(self.json_repair_adapter.model_name if (reasoner_repaired and self.json_repair_adapter) else None),
                repair_error=reasoner_repair_error,
            )

            if reasoner_resp.error or not reasoning_json:
                error = reasoner_resp.error or reasoner_repair_error or "reasoner failed"
                break

            candidate_answer = ""
            if isinstance(reasoning_json, dict):
                candidate = reasoning_json.get("final_answer", "")
                if isinstance(candidate, str):
                    candidate_answer = extract_answer(candidate)
            if not candidate_answer:
                candidate_answer = extract_answer(reasoner_text)
            last_candidate_answer = candidate_answer

            critic_prompt = self.critic.build_prompt(question_text, planner_json, reasoning_json if isinstance(reasoning_json, dict) else {})
            critic_resp = await self.adapter.complete(critic_prompt)
            critic_text = critic_resp.content if critic_resp.success else ""
            critic_json, critic_repaired, critic_repair_error = await self._parse_or_repair(
                "critic",
                critic_text,
                {"verdict": "approve|reject", "issues": ["..."], "required_fixes": "..."},
            )
            critic_out = AgentOutput(
                prompt=critic_prompt,
                response=critic_text,
                parsed=critic_json or None,
                latency_ms=critic_resp.latency_ms,
                error=critic_resp.error,
                repaired=critic_repaired,
                repair_model=(self.json_repair_adapter.model_name if (critic_repaired and self.json_repair_adapter) else None),
                repair_error=critic_repair_error,
            )

            if critic_resp.error or not critic_json:
                error = critic_resp.error or critic_repair_error or "critic failed"
                break

            verdict = critic_json.get("verdict") if isinstance(critic_json, dict) else None
            if verdict == "approve" and candidate_answer:
                break

            required_fixes = ""
            if isinstance(critic_json, dict):
                required_fixes = str(critic_json.get("required_fixes", "") or "")
            critic_feedback = required_fixes.strip()

        if not reasoner_out:
            reasoner_out = AgentOutput(prompt="", response="", parsed=None)
        final_answer = "" if error else last_candidate_answer

        latency_ms = int((time.time() - start_time) * 1000)
        return AgenticSolveResult(
            final_answer=final_answer,
            planner=planner_out,
            reasoner=reasoner_out,
            critic=critic_out,
            rounds=rounds_used,
            error=error,
            latency_ms=latency_ms,
        )
