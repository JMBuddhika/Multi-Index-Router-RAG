from __future__ import annotations
from typing import List, Dict, Any
import os
import duckdb
from ..llm_groq import GroqLLM

class DuckSQL:
    def __init__(self, db_path: str = ":memory:"):
        self.con = duckdb.connect(db_path)
        self.llm = GroqLLM()

    def ingest_csv_glob(self, folder: str):
        import glob
        csvs = glob.glob(os.path.join(folder, "*.csv"))
        for p in csvs:
            name = os.path.splitext(os.path.basename(p))[0]
            self.con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_csv_auto('{p}', header=True)")
        return self.list_tables()

    def list_tables(self) -> List[Dict[str, Any]]:
        rows = self.con.execute("SHOW TABLES").fetchall()
        out = []
        for (name,) in rows:
            cols = self.con.execute(f"DESCRIBE {name}").fetchall()
            schema = [{"name": c[0], "type": c[1]} for c in cols]
            out.append({"table": name, "schema": schema})
        return out

    def _schema_text(self) -> str:
        info = self.list_tables()
        lines = []
        for t in info:
            cols = ", ".join([f"{c['name']} {c['type']}" for c in t["schema"]])
            lines.append(f"TABLE {t['table']}({cols})")
        return "\n".join(lines) if lines else "No tables."

    def text2sql(self, question: str) -> str:
        system = (
            "You convert questions into a single ANSI SQL query for DuckDB.\n"
            "Only output SQL. Use table and column names exactly as provided.\n"
            "Prefer simple SELECTs. Limit 100 rows."
        )
        schema = self._schema_text()
        user = f"Schema:\n{schema}\n\nQuestion: {question}\nSQL:"
        sql = self.llm.chat(system, user, temperature=0.0)
        sql = sql.strip().strip("```").replace("sql\n", "").replace("SQL\n", "").strip()
        if not sql.lower().lstrip().startswith("select"):
            sql = f"SELECT 'Unable to derive SQL for: {question}' as note"
        return sql

    def query(self, question: str) -> Dict[str, Any]:
        sql = self.text2sql(question)
        try:
            rows = self.con.execute(sql).fetchall()
            cols = [d[0] for d in self.con.description] if self.con.description else []
            return {"sql": sql, "columns": cols, "rows": rows}
        except Exception as e:
            return {"sql": sql, "error": str(e), "columns": [], "rows": []}
