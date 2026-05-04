from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import mysql.connector
import requests
import json
import re
import os

app = Flask(__name__, static_folder=".")
CORS(app)

# ── Config ────────────────────────────────────────────────────────────────────
LLAMA_URL   = "http://localhost:8081/v1/chat/completions"   # your llama-server
DB_CONFIG   = {
    "host":     "localhost",
    "user":     "root",          # change to your MySQL user
    "password": "yourpassword",  # change to your MySQL password
    "database": "yourdb",        # change to your database name
}
# ─────────────────────────────────────────────────────────────────────────────


def get_db_schema():
    """Fetch table names and columns from the connected MySQL DB."""
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SHOW TABLES")
    tables = [row[0] for row in cursor.fetchall()]
    schema_parts = []
    for table in tables:
        cursor.execute(f"DESCRIBE `{table}`")
        cols = [f"  {row[0]} ({row[1]})" for row in cursor.fetchall()]
        schema_parts.append(f"Table `{table}`:\n" + "\n".join(cols))
    cursor.close()
    conn.close()
    return "\n\n".join(schema_parts)


def ask_llama(system_prompt, user_message):
    """Send a chat completion request to the local llama-server."""
    payload = {
        "model": "bonsai",          # llama-server ignores this in router mode
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        "temperature": 0.1,
        "max_tokens": 512,
    }
    resp = requests.post(LLAMA_URL, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def extract_sql(text):
    """Pull the first SQL statement out of a fenced or plain response."""
    # try ```sql ... ``` block first
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # fallback: grab from SELECT/INSERT/UPDATE/DELETE onward
    m = re.search(r"(SELECT|INSERT|UPDATE|DELETE|WITH)\b.*", text,
                  re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(0).strip()
    return None


def run_sql(sql):
    """Execute a SQL query and return (columns, rows) or raise on error."""
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute(sql)
    columns = [d[0] for d in cursor.description] if cursor.description else []
    rows    = [list(r) for r in cursor.fetchall()]
    cursor.close()
    conn.close()
    return columns, rows


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/schema")
def schema():
    try:
        return jsonify({"schema": get_db_schema()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    user_question = data.get("question", "").strip()
    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    # Step 1 — get schema
    try:
        schema_text = get_db_schema()
    except Exception as e:
        return jsonify({"error": f"DB connection failed: {e}"}), 500

    # Step 2 — ask LLM to generate SQL
    sql_system = (
        "You are an expert SQL assistant. "
        "Given the database schema below and a user question, "
        "write a single valid MySQL SELECT query that answers it. "
        "Return ONLY the SQL query inside a ```sql block, nothing else.\n\n"
        f"Schema:\n{schema_text}"
    )
    try:
        sql_response = ask_llama(sql_system, user_question)
    except Exception as e:
        return jsonify({"error": f"LLM (SQL generation) failed: {e}"}), 500

    sql = extract_sql(sql_response)
    if not sql:
        return jsonify({
            "error": "Could not extract a SQL query from the model response.",
            "raw":   sql_response,
        }), 422

    # Step 3 — run the SQL
    try:
        columns, rows = run_sql(sql)
    except Exception as e:
        return jsonify({"error": f"SQL execution failed: {e}", "sql": sql}), 422

    # Step 4 — ask LLM to explain the results
    results_text = json.dumps({"columns": columns, "rows": rows[:20]}, indent=2)
    explain_system = (
        "You are a helpful data analyst. "
        "Explain the following query results in clear, plain English. "
        "Be concise (3-5 sentences). Mention key numbers or trends."
    )
    explain_prompt = (
        f"User asked: {user_question}\n\n"
        f"SQL used:\n{sql}\n\n"
        f"Results (first 20 rows):\n{results_text}"
    )
    try:
        explanation = ask_llama(explain_system, explain_prompt)
    except Exception as e:
        explanation = f"(Explanation unavailable: {e})"

    return jsonify({
        "question":    user_question,
        "sql":         sql,
        "columns":     columns,
        "rows":        rows,
        "explanation": explanation,
    })


if __name__ == "__main__":
    print("Starting DB-LLM bridge on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
