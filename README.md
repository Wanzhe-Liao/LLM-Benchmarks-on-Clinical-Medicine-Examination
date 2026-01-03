# LLM CMCA评测  BEYOND ACCURACY

## 快速开始

1) 安装依赖

```bash
pip install -r requirements.txt
```

2) 配置模型

- 复制 `config.example.yaml` 为 `config.yaml`
- 设置环境变量 `OPENAI_API_KEY`（或直接在 `config.yaml` 填写 `api_key`）

> 配置里支持 `${ENV_VAR}` 形式：会在运行时从环境变量读取；若环境变量不存在则视为空。

3) 准备题库

- 参考 `data/example_exam.md` 的格式（必须包含 `## 一、A型题` / `## 二、B型题` / `## 三、X型题` / `## 参考答案`）
- 建议不要提交、传播真实考试题目与答案, 所有题目仅用于科学研究

4) 运行

```bash
python main.py --config config.yaml --exam data/example_exam.md --mode baseline --limit 4
```

输出默认在 `results/YYYYMMDD_HHMMSS/`。

