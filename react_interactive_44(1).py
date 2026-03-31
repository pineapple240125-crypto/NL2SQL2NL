import os
import re
import json
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from openai import OpenAI
import numpy as np
# 新增：从visualization模块引入visualize_result函数
from visualization import visualize_result

# ================== 基础配置 ==================
DB_PATH = "data.db"
SCHEMA_PATH = "schema.json"
TOKEN_FILE = "llm_token"  # 存储API密钥的文件
MODEL = ""

# Ensemble配置 - 新增
ENSEMBLE_SETTINGS = {
    "num_variants": 3,  # 生成多少个SQL变体
    "temperature_range": [0.1, 0.7],  # 温度变化范围
    "voting_threshold": 2,  # 投票阈值
    "enable_self_correction": True  # 是否启用自我修正
}


# 从文件读取API密钥
def load_api_key():
    if not os.path.exists(TOKEN_FILE):
        raise FileNotFoundError(f"未找到密钥文件：{TOKEN_FILE}，请创建该文件并填入ARK_API_KEY")
    with open(TOKEN_FILE, "r") as f:
        key = f.read().strip()  # 读取并去除首尾空格/换行
    if not key:
        raise ValueError(f"密钥文件 {TOKEN_FILE} 内容为空，请填入有效的ARK_API_KEY")
    return key


# 初始化客户端（使用文件中的密钥）
client = OpenAI(
    base_url="",
    api_key=load_api_key()  # 从文件加载密钥
)


# ================== Step 1: CSV → SQLite ==================
def csv_to_sqlite(csv_path: str, table_name: str, db_path: str = DB_PATH, if_exists: str = "replace"):
    """
    将 CSV 文件写入 SQLite 数据库，并保存 Schema 信息到 schema.json
    """
    df = pd.read_csv(csv_path)
    df.columns = [c if c and str(c).strip() else f"col_{i}" for i, c in enumerate(df.columns)]

    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists=if_exists, index=False)
    conn.close()

    schema_info = {
        "table_name": table_name,
        "columns": {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}
    }

    if os.path.exists(SCHEMA_PATH):
        with open(SCHEMA_PATH, "r") as f:
            existing = json.load(f)
    else:
        existing = {}

    existing[table_name] = schema_info
    with open(SCHEMA_PATH, "w") as f:
        json.dump(existing, f, indent=4, ensure_ascii=False)

    print(f"\n✅ [INFO] 已导入 CSV: {csv_path} -> SQLite 表 {table_name}")
    print(f"[INFO] Schema 已保存到 {SCHEMA_PATH}")
    return schema_info


# ================== Step 2: SQL 执行 & 可视化 ==================
def run_sql(query: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        col_names = [desc[0] for desc in cursor.description]
        return {"columns": col_names, "rows": result}
    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.close()


# ================== 单模型Ensemble模块 - 新增 ==================
class SingleModelEnsemble:
    def __init__(self, client, schema: dict):
        self.client = client
        self.schema = schema

    def generate_sql_with_ensemble(self, query: str) -> str:
        """
        使用单模型生成多个SQL变体，并进行投票融合
        """
        print(f"开始生成 {ENSEMBLE_SETTINGS['num_variants']} 个SQL变体...")

        # 步骤1: 生成多个SQL变体
        sql_variants = self._generate_sql_variants(query)

        # 步骤2: 验证和评分每个变体
        validated_variants = self._validate_sql_variants(sql_variants, query)

        # 步骤3: 投票选择最佳SQL
        best_sql = self._vote_best_sql(validated_variants)

        # 步骤4: 自我修正（可选）
        if ENSEMBLE_SETTINGS["enable_self_correction"]:
            best_sql = self._self_correct_sql(best_sql, query, validated_variants)

        # 记录ensemble过程
        self._log_ensemble_process(query, sql_variants, validated_variants, best_sql)

        print(f"最终选择SQL: {best_sql}")
        return best_sql

    def _generate_sql_variants(self, query: str) -> list:
        """生成多个SQL变体"""
        variants = []
        table = self.schema["table_name"]
        cols = ", ".join(self.schema["columns"].keys())

        # 不同的提示策略
        prompt_strategies = [
            # 策略1: 标准提示
            f"""
            You are an expert SQL generator. Output only SQL.
            Table: {table}
            Columns: {cols}

            Convert to accurate SQLite SQL:
            Input: {query}
            Output: (SQL only)
            """,

            # 策略2: 思维链提示
            f"""
            Think step by step and generate SQL:
            1. Analyze the user query: "{query}"
            2. Consider table structure: {table} with columns: {cols}
            3. Generate precise SQLite SQL

            Output only the final SQL statement.
            """,

            # 策略3: 示例引导提示
            f"""
            Examples of good SQL generation:
            Input: "show average salary per department"
            Output: SELECT department, AVG(salary) FROM {table} GROUP BY department;

            Input: "count cities in each country"  
            Output: SELECT country, COUNT(*) AS city_count FROM {table} GROUP BY country;

            Now generate SQL for:
            Input: {query}
            Output: (SQL only)
            """
        ]

        # 温度变化生成不同变体
        temperatures = np.linspace(
            ENSEMBLE_SETTINGS["temperature_range"][0],
            ENSEMBLE_SETTINGS["temperature_range"][1],
            ENSEMBLE_SETTINGS["num_variants"]
        )

        for i, temp in enumerate(temperatures):
            # 轮换使用不同的提示策略
            prompt_index = i % len(prompt_strategies)
            prompt = prompt_strategies[prompt_index]

            try:
                response = self.client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temp,
                )

                sql = response.choices[0].message.content.strip()
                sql = re.sub(r"^```sql|```$", "", sql, flags=re.IGNORECASE).strip()

                variants.append({
                    "variant_id": i + 1,
                    "sql": sql,
                    "temperature": temp,
                    "prompt_strategy": prompt_index
                })

                print(f"✅ 变体 {i + 1} 生成完成 (temp={temp:.2f}): {sql}")

            except Exception as e:
                print(f"❌ 变体 {i + 1} 生成失败: {e}")

        return variants

    def _validate_sql_variants(self, variants: list, original_query: str) -> list:
        """验证和评分每个SQL变体"""
        validated_variants = []

        for variant in variants:
            sql = variant["sql"]

            # 综合评分（0-100分）
            score = 0

            # 1. 语法验证 (30分)
            syntax_score = self._validate_syntax(sql) * 30
            score += syntax_score

            # 2. 语义验证 (40分)
            semantic_score = self._validate_semantics(sql, original_query) * 40
            score += semantic_score

            # 3. 结构合理性验证 (30分)
            structure_score = self._validate_structure(sql) * 30
            score += structure_score

            variant.update({
                "syntax_score": syntax_score,
                "semantic_score": semantic_score,
                "structure_score": structure_score,
                "total_score": score,
                "is_executable": syntax_score > 20,
                "validation_passed": score > 60
            })

            validated_variants.append(variant)

        # 按总分排序
        validated_variants.sort(key=lambda x: x["total_score"], reverse=True)
        return validated_variants

    def _validate_syntax(self, sql: str) -> float:
        """语法验证（0-1分）"""
        try:
            # 基础语法检查
            if not sql.upper().startswith(('SELECT', 'WITH')):
                return 0.0

            # 检查基本SQL结构
            if 'FROM' not in sql.upper():
                return 0.3

            # 尝试解析
            conn = sqlite3.connect(':memory:')
            try:
                conn.execute(f"EXPLAIN {sql}")
                return 1.0
            except:
                if len(sql.split()) >= 4:
                    return 0.7
                return 0.5
            finally:
                conn.close()

        except:
            return 0.0

    def _validate_semantics(self, sql: str, original_query: str) -> float:
        """语义验证（0-1分）"""
        score = 0.0

        # 1. 查询意图匹配
        query_keywords = set(re.findall(r'\b\w+\b', original_query.lower()))
        sql_keywords = set(re.findall(r'\b\w+\b', sql.lower()))

        matching_keywords = query_keywords.intersection(sql_keywords)
        if matching_keywords:
            score += 0.4 * (len(matching_keywords) / len(query_keywords))

        # 2. 逻辑合理性检查
        if any(op in sql.upper() for op in ['GROUP BY', 'ORDER BY', 'WHERE']):
            score += 0.3

        # 3. 结果列合理性
        if 'SELECT' in sql.upper() and 'FROM' in sql.upper():
            score += 0.3

        return min(score, 1.0)

    def _validate_structure(self, sql: str) -> float:
        """结构合理性验证（0-1分）"""
        score = 0.0

        sql_upper = sql.upper()

        # 基本结构分
        if 'SELECT' in sql_upper and 'FROM' in sql_upper:
            score += 0.4

        # 子句完整性
        clauses = ['WHERE', 'GROUP BY', 'HAVING', 'ORDER BY']
        present_clauses = [clause for clause in clauses if clause in sql_upper]
        if present_clauses:
            score += 0.3 * (len(present_clauses) / len(clauses))

        # 结束合理性
        if not sql_upper.endswith(';') or sql_upper.count(';') == 1:
            score += 0.3

        return score

    def _vote_best_sql(self, validated_variants: list) -> str:
        """投票选择最佳SQL"""
        if not validated_variants:
            raise ValueError("没有可用的SQL变体")

        # 过滤掉验证失败的变体
        valid_variants = [v for v in validated_variants if v["validation_passed"]]

        if not valid_variants:
            # 如果没有完全通过的，选择分数最高的
            best_variant = max(validated_variants, key=lambda x: x["total_score"])
            print(f"⚠️无完美变体，选择最高分: {best_variant['total_score']:.1f}")
            return best_variant["sql"]

        # 选择验证通过的变体中分数最高的
        best_variant = max(valid_variants, key=lambda x: x["total_score"])

        print(f"选择最佳变体，分数: {best_variant['total_score']:.1f}")

        return best_variant["sql"]

    def _self_correct_sql(self, best_sql: str, query: str, variants: list) -> str:
        """自我修正：基于多个变体改进SQL"""
        if len(variants) < 2:
            return best_sql

        # 收集所有变体的优点
        good_parts = []
        for variant in variants[:3]:
            if variant["total_score"] > 50:
                good_parts.append(variant["sql"])

        if len(good_parts) < 2:
            return best_sql

        # 使用模型进行自我修正
        correction_prompt = f"""
        基于以下SQL变体，生成一个改进版本：

        原始查询: {query}

        变体1: {good_parts[0]}
        变体2: {good_parts[1]}
        {f'变体3: {good_parts[2]}' if len(good_parts) > 2 else ''}

        请分析各变体的优点，生成一个更准确、更完整的SQL语句。
        只输出最终的SQL语句。
        """

        try:
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": correction_prompt}],
                temperature=0.1,
            )

            corrected_sql = response.choices[0].message.content.strip()
            corrected_sql = re.sub(r"^```sql|```$", "", corrected_sql, flags=re.IGNORECASE).strip()

            # 验证修正后的SQL
            corrected_score = self._validate_syntax(corrected_sql) * 30 + \
                              self._validate_semantics(corrected_sql, query) * 40 + \
                              self._validate_structure(corrected_sql) * 30

            original_score = next((v["total_score"] for v in variants if v["sql"] == best_sql), 0)

            if corrected_score > original_score:
                print(f"自我修正完成，分数提升: {original_score:.1f} → {corrected_score:.1f}")
                return corrected_sql
            else:
                print(f"自我修正未提升分数，保持原SQL")
                return best_sql

        except Exception as e:
            print(f"⚠自我修正失败: {e}")
            return best_sql

    def _log_ensemble_process(self, query: str, variants: list, validated_variants: list, best_sql: str):
        """记录Ensemble决策过程"""
        log_entry = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "query": query,
            "best_sql": best_sql,
            "variants_generated": len(variants),
            "variants_passed": len([v for v in validated_variants if v["validation_passed"]]),
            "variant_details": validated_variants
        }

        # 保存到日志文件
        os.makedirs("ensemble_logs", exist_ok=True)
        log_file = os.path.join("ensemble_logs", "single_model_ensemble.jsonl")

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


# ================== Step 3: NL → SQL（修改为使用Ensemble） ==================
def nl_to_sql(query: str, schema: dict) -> str:
    """
    使用单模型Ensemble将自然语言转为SQL
    """
    ensemble = SingleModelEnsemble(client, schema)
    return ensemble.generate_sql_with_ensemble(query)


# ================== Step 4: ReAct Agent ==================
def parse_react_output(text: str):
    thought_match = re.search(r"Thought:\s*(.*)", text)
    action_match = re.search(r'Action:\s*([A-Za-z_]+)\["(.*?)"\]', text)
    final_match = re.search(r'Final Answer\["(.*?)"\]', text)

    thought = thought_match.group(1).strip() if thought_match else ""
    action = action_match.groups() if action_match else None
    final = final_match.group(1).strip() if final_match else None
    return {"thought": thought, "action": action, "final": final, "raw": text}


def react_agent(task: str, schema: dict, verbose=True):
    """ReAct Agent 主循环（全部使用Ensemble）"""
    history = f"user query: {task}\n"
    obs = ""
    max_steps = 6
    step = 0

    while step < max_steps:
        step += 1
        prompt = f"""{history}
                        You are a ReAct agent that follows the format below exactly.
                        Format:
                        Thought: <one reasoning step>
                        Action: <ActionName>["argument"]
                        Action choice:
- GenerateSQL["user natural language"]  
  -> Put here the user's task in plain natural language that should be converted to SQL (NOT SQL).  
  Example (correct): GenerateSQL["show total population per country"]  
  Example (incorrect): GenerateSQL["SELECT country, SUM(pop) FROM city GROUP BY country;"]
- RunSQL["SQL query"]  
  -> An executable SQLite SELECT query. Example: RunSQL["SELECT country, SUM(population) AS tot FROM city GROUP BY country;"]
- Visualize["pie"/"bar"/"line"]  
  -> Visualize the most recent SQL result saved by the system.
- Final Answer["..."]  
  -> Finish and produce the final textual response. Only use Final Answer after you have used the tools and have an Observation containing the final analysis or chart path.
                        Rules:
                        - Use the last Observation as the result of your previous Action.
                        - Continue reasoning from there.
                        - Respond with exactly one Thought（in one concise sentence） and one Action.
                        - If you have the final answer, use Final Answer["..."] as the Action.
                        - All SQL generation uses single-model ensemble for better accuracy.
                        Think step by step.
Example Interaction:
Final Answer:"Bar chart saved as plot.png shows average salary per department."
User query: Show average salary per department.
Thought: I need to convert the natural language to SQL using ensemble for better accuracy.
Action: GenerateSQL["average salary per department"]
Observation: SQL generated using ensemble: SELECT department, AVG(salary) FROM employees GROUP BY department;
Thought: I should run this SQL to see the results.
Action: RunSQL["SELECT department, AVG(salary) FROM employees GROUP BY department;"]
Observation: SQL executed successfully.
Thought: I want to visualize it.
Action: Visualize["bar"]
Observation: Visualization saved as plot.png
                        """
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        output = response.choices[0].message.content.strip()
        parsed = parse_react_output(output)
        # if verbose:
            # print(f"\n--- Step {step} Output ---\n{output}")

        if parsed["action"]:
            name, arg = parsed["action"]

            if name == "GenerateSQL":
                sql = nl_to_sql(arg, schema)
                obs = f"SQL generated using ensemble: {sql}"
                history += f"{output}\nObservation: {obs}\n"

            elif name == "RunSQL":
                # current SQL 和重试计数的处理（history 是字符串）
                current_sql = arg.strip()
                # 从 history 字符串中查找是否已有重试记录，格式为: __retry__:<sql>:<count>\n
                m = re.search(rf"__retry__:{re.escape(current_sql)}:(\d+)\n", history)
                retry_count = int(m.group(1)) if m else 0

                result = run_sql(arg)
                if "error" in result:
                    err_msg = result["error"]
                    retry_count += 1
                    obs = f"Error executing SQL: {err_msg}\nSQL was: {current_sql}"
                    obs += ("\nPlease fix the SQL and respond with GenerateSQL[\"<fixed natural language query>\"] "
                            "containing the natural language query for ensemble generation.")

                    # 更新 history 中的重试记录
                    history = re.sub(rf"__retry__:{re.escape(current_sql)}:\d+\n", "", history)
                    history += f"__retry__:{current_sql}:{retry_count}\n"
                    MAX_SQL_RETRIES = 2
                    if retry_count > MAX_SQL_RETRIES:
                        obs += f"\nMaximum retry limit ({MAX_SQL_RETRIES}) reached for this SQL. Aborting this SQL."
                        history = re.sub(rf"__retry__:{re.escape(current_sql)}:\d+\n", "", history)

                else:
                    df = pd.DataFrame(result["rows"], columns=result["columns"]).head(10)
                    df.to_csv("temp_result.csv", index=False)
                    obs = f"SQL executed, got {len(df)} rows. Saved to temp_result.csv"
                    print(obs)
                    history = re.sub(rf"__retry__:{re.escape(current_sql)}:\d+\n", "", history)
                history += f"{output}\nObservation: {obs}\n"

            elif name == "Visualize":
                print(f"[验证] Visualize被调用，参数: {arg}，当前步骤: {step}")
                if not os.path.exists("temp_result.csv"):
                    obs = "Error: no SQL result to visualize"
                else:
                    df = pd.read_csv("temp_result.csv")
                    # 修改：适配从visualization导入的函数参数格式
                    # 创建唯一报告目录（使用UUID确保唯一性）
                    import uuid  # 局部导入，避免全局依赖
                    report_dir = os.path.join("visual_reports", f"report_{uuid.uuid4().hex[:8]}")
                    path = visualize_result(
                        df=df,
                        chart_type_str=arg,  # 对应visualization模块的参数名
                        output_dir=report_dir,
                        auto_open=True  # 自动打开生成的报告
                    )
                    obs = f"Visualization saved to {path}"
                history += f"{output}\nObservation: {obs}\n"

            else:
                obs = f"Unknown Action: {name}"

        elif parsed["final"]:
            print(f"\n✅ Final Answer: {parsed['final']}")
            return history
            break

        else:
            obs = "No valid action detected."
        history += f"{output}\nObservation: {obs}\n"


# ================== 交互式入口 ==================
if __name__ == "__main__":
    print("=== 🧠🧠🧠 ReAct 数据分析智能体 ===\n")

    csv_path = input("请输入 CSV 文件路径（如 city.csv）: ").strip()
    table_name = os.path.splitext(os.path.basename(csv_path))[0]
    schema = csv_to_sqlite(csv_path, table_name)

    while True:
        query = input("\n请输入你的自然语言问题（或输入 exit 退出）:\n> ").strip()
        if query.lower() in ["exit", "quit"]:
            print("👋👋 再见！")
            break
        print("🚀🚀 ReAct 数据分析开始啦，准备好迎接答案了吗？...")
        trace = react_agent(query, schema)
        print(trace)