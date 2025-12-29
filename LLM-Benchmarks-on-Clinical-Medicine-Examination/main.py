import argparse
import asyncio
from datetime import datetime
import os
import re

import yaml

from exam.parser import ExamParser
from models.factory import ModelFactory
from report.generator import ReportGenerator
from runner.batch_runner import BatchRunner


_ENV_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


def _resolve_env(value):
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    if isinstance(value, str):
        m = _ENV_PATTERN.match(value.strip())
        if m:
            return os.environ.get(m.group(1), "")
    return value


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return _resolve_env(cfg)


def print_progress(completed: int, total: int, model_id: str, q_id: int, mode: str):
    percent = completed / total * 100 if total else 0
    print(f"\r进度: {completed}/{total} ({percent:.1f}%) - {mode} - {model_id} Q{q_id}", end="", flush=True)


async def run_one_mode(adapters, questions, max_concurrent: int, agentic_max_rounds: int, mode: str, json_repair_adapter=None):
    runner = BatchRunner(
        adapters,
        max_concurrent=max_concurrent,
        agentic_max_rounds=agentic_max_rounds,
        json_repair_adapter=json_repair_adapter,
    )

    def cb(c, t, m, q):
        print_progress(c, t, m, q, mode)

    return await runner.run_all_models(questions, mode=mode, progress_callback=cb)


async def run_tests(
    config: dict,
    exam_path: str,
    output_dir: str,
    models_filter: list | None = None,
    runs: int = 1,
    mode: str = "both",
    limit: int | None = None,
    question_ids: list[int] | None = None,
):
    print("解析题目文件...")
    parser = ExamParser(exam_path)
    questions = parser.parse()
    if question_ids:
        wanted = set(question_ids)
        questions = [q for q in questions if q.id in wanted]
    if limit is not None:
        questions = questions[: max(0, int(limit))]
    print(f"共解析 {len(questions)} 道题")

    test_config = config.get("test", {}) or {}
    models_config = config.get("models", {}) or {}
    if models_filter:
        models_config = {k: v for k, v in models_config.items() if k in models_filter}

    print(f"初始化 {len(models_config)} 个模型适配器...")
    merged_models_config = {}
    for model_id, model_cfg in models_config.items():
        cfg = dict(model_cfg or {})
        cfg["timeout"] = test_config.get("timeout", cfg.get("timeout", 120))
        cfg["temperature"] = test_config.get("temperature", cfg.get("temperature", 0.0))
        max_tokens = test_config.get("max_tokens", cfg.get("max_tokens", 1024))
        try:
            max_tokens = int(max_tokens)
        except Exception:
            max_tokens = 1024
        cfg["max_tokens"] = min(max_tokens, 10000)
        merged_models_config[model_id] = cfg

    adapters = ModelFactory.create_all(merged_models_config)
    if not adapters:
        print("错误: 没有配置有效的模型(需要填写 api_key 和 base_url)")
        return

    json_repair_adapter = None
    json_repair_cfg = config.get("json_repair") or {}
    if json_repair_cfg.get("api_key") and json_repair_cfg.get("base_url") and json_repair_cfg.get("model_name"):
        repair_cfg = dict(json_repair_cfg)
        repair_cfg["timeout"] = int(repair_cfg.get("timeout") or test_config.get("timeout", 120))
        repair_cfg["temperature"] = float(repair_cfg.get("temperature") if repair_cfg.get("temperature") is not None else test_config.get("temperature", 0.0))
        max_tokens = repair_cfg.get("max_tokens", 800)
        try:
            max_tokens = int(max_tokens)
        except Exception:
            max_tokens = 800
        repair_cfg["max_tokens"] = min(max(200, max_tokens), 2000)
        json_repair_adapter = ModelFactory.create("__json_repair__", repair_cfg)

    max_concurrent = int(test_config.get("max_concurrent", 5))
    agentic_max_rounds = int(test_config.get("agentic_max_rounds", 2))

    mode = mode.lower().strip()
    if mode not in ("baseline", "agentic", "both"):
        raise ValueError("--mode must be baseline|agentic|both")

    all_runs: list[dict] = []
    generator = ReportGenerator(questions, output_dir)

    for run_idx in range(runs):
        print(f"\n========== 第 {run_idx + 1}/{runs} 轮 ==========")

        baseline_results = {}
        if mode in ("baseline", "both"):
            baseline_results = await run_one_mode(adapters, questions, max_concurrent, agentic_max_rounds, "baseline", json_repair_adapter=json_repair_adapter)
            print("\nBaseline 完成")
        else:
            print("\nBaseline 已跳过（由外部单独测试）")

        agentic_results = None
        if mode in ("agentic", "both"):
            agentic_results = await run_one_mode(adapters, questions, max_concurrent, agentic_max_rounds, "agentic", json_repair_adapter=json_repair_adapter)
            print("\nAgentic 完成")

        all_runs.append({"baseline": baseline_results, "agentic": agentic_results})

        if runs > 1:
            generator.save_multi_run_results(all_runs)
            generator.generate_multi_run_report(all_runs)
            print(f"\n已保存中间结果: {run_idx + 1}/{runs} 轮")

    print("\n========== 生成报告 ==========")
    if runs == 1:
        report_path = generator.generate_report(all_runs[0]["baseline"], all_runs[0]["agentic"])
        generator.save_results(all_runs[0]["baseline"], all_runs[0]["agentic"])
    else:
        report_path = generator.generate_multi_run_report(all_runs)
        generator.save_multi_run_results(all_runs)

    print(f"报告已生成: {report_path}")
    print(f"数据已保存于: {output_dir}")


def main():
    arg_parser = argparse.ArgumentParser(description="LLM 医学考试测试平台（禁用网络搜索）")
    arg_parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    arg_parser.add_argument("--exam", default="data/example_exam.md", help="题目文件路径")
    arg_parser.add_argument("--output", default=None, help="输出目录")
    arg_parser.add_argument("--models", nargs="+", help="指定测试的模型")
    arg_parser.add_argument("--runs", type=int, default=1, help="重复运行次数（用于稳定性统计）")
    arg_parser.add_argument("--mode", choices=["baseline", "agentic", "both"], default="both", help="运行模式")
    arg_parser.add_argument("--limit", type=int, default=None, help="仅运行前 N 题（用于快速验证）")
    arg_parser.add_argument("--question-ids", nargs="+", type=int, default=None, help="仅运行指定题号（空格分隔）")

    args = arg_parser.parse_args()
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/{timestamp}"

    config = load_config(args.config)

    asyncio.run(
        run_tests(
            config=config,
            exam_path=args.exam,
            output_dir=args.output,
            models_filter=args.models,
            runs=args.runs,
            mode=args.mode,
            limit=args.limit,
            question_ids=args.question_ids,
        )
    )


if __name__ == "__main__":
    main()
