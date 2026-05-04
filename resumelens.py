import streamlit as st
import mysql.connector
import requests
import json
import re

st.set_page_config(page_title="ResumeLens · DB Query", page_icon="🔍", layout="wide")

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: #0d1117;
    color: #e6edf3;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
.block-container { padding: 2rem 2.5rem !important; }

.title {
    text-align: center;
    font-size: 2rem;
    font-weight: bold;
    color: #58a6ff;
    margin-bottom: 0.2rem;
}
.subtitle {
    text-align: center;
    color: #8b949e;
    font-size: 0.85rem;
    margin-bottom: 2rem;
}
.sql-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1rem;
    font-family: monospace;
    font-size: 0.85rem;
    color: #79c0ff;
    white-space: pre-wrap;
    margin-bottom: 1rem;
}
.explain-box {
    background: #0f2a1a;
    border: 1px solid #238636;
    border-radius: 8px;
    padding: 1rem;
    color: #aff5b4;
    font-size: 0.95rem;
    margin-bottom: 1rem;
}
.stButton > button {
    background: #238636 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: bold !important;
    padding: 0.5rem 1.5rem !important;
}
.stButton > button:hover {
    background: #2ea043 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────────────────────
LLAMA_URL = "http://localhost:8081/v1/chat/completions"

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="title">🔍 ResumeLens · DB Query</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask questions about your database in plain English</div>', unsafe_allow_html=True)

# ── Sidebar — DB Config ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Database Config")
    db_host = st.text_input("Host",     value="localhost")
    db_user = st.text_input("User",     value="root")
    db_pass = st.text_input("Password", value="", type="password")
    db_name = st.text_input("Database", value="")
    st.markdown("---")
    st.markdown("### 🤖 Ollama Config")
    llama_url = st.text_input("Ollama URL", value=LLAMA_URL)

DB_CONFIG = {
    "host":     db_host,
    "user":     db_user,
    "password": db_pass,
    "database": db_name,
}

# ── Helper functions ──────────────────────────────────────────────────────────
def get_db_schema():
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
    payload = {
        "model": "bonsai",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        "temperature": 0.1,
        "max_tokens": 512,
    }
    resp = requests.post(llama_url, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()

def extract_sql(text):
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"(SELECT|INSERT|UPDATE|DELETE|WITH)\b.*", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(0).strip()
    return None

def run_sql(sql):
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute(sql)
    columns = [d[0] for d in cursor.description] if cursor.description else []
    rows    = [list(r) for r in cursor.fetchall()]
    cursor.close()
    conn.close()
    return columns, rows

# ── Schema viewer ─────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col2:
    if st.button("📋 Show Schema", use_container_width=True):
        try:
            schema = get_db_schema()
            st.code(schema, language="sql")
        except Exception as e:
            st.error(f"DB connection failed: {e}")

# ── Query input ───────────────────────────────────────────────────────────────
with col1:
    question = st.text_input(
        "Ask a question about your database",
        placeholder="e.g. How many users signed up last month?",
        label_visibility="collapsed"
    )

# ── Run query ─────────────────────────────────────────────────────────────────
if st.button("🔍 Run Query", use_container_width=False):
    if not question.strip():
        st.warning("Please enter a question!")
    elif not db_name:
        st.warning("Please enter your database name in the sidebar!")
    else:
        with st.spinner("Thinking..."):
            try:
                # Step 1 — get schema
                schema_text = get_db_schema()

                # Step 2 — generate SQL
                sql_system = (
                    "You are an expert SQL assistant. "
                    "Given the database schema below and a user question, "
                    "write a single valid MySQL SELECT query that answers it. "
                    "Return ONLY the SQL query inside a ```sql block, nothing else.\n\n"
                    f"Schema:\n{schema_text}"
                )
                sql_response = ask_llama(sql_system, question)
                sql = extract_sql(sql_response)

                if not sql:
                    st.error("Could not extract SQL from model response.")
                    st.code(sql_response)
                else:
                    # Step 3 — run SQL
                    st.markdown("**Generated SQL:**")
                    st.markdown(f'<div class="sql-box">{sql}</div>', unsafe_allow_html=True)

                    columns, rows = run_sql(sql)

                    # Step 4 — explain results
                    results_text = json.dumps({"columns": columns, "rows": rows[:20]}, indent=2)
                    explain_system = (
                        "You are a helpful data analyst. "
                        "Explain the following query results in clear, plain English. "
                        "Be concise (3-5 sentences). Mention key numbers or trends."
                    )
                    explain_prompt = (
                        f"User asked: {question}\n\n"
                        f"SQL used:\n{sql}\n\n"
                        f"Results (first 20 rows):\n{results_text}"
                    )
                    explanation = ask_llama(explain_system, explain_prompt)

                    st.markdown("**Explanation:**")
                    st.markdown(f'<div class="explain-box">{explanation}</div>', unsafe_allow_html=True)

                    st.markdown("**Results:**")
                    if rows:
                        import pandas as pd
                        df = pd.DataFrame(rows, columns=columns)
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("No results returned.")

            except mysql.connector.Error as e:
                st.error(f"❌ Database error: {e}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to Ollama. Make sure it's running on port 8081.")
            except Exception as e:
                st.error(f"❌ Error: {e}")