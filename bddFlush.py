import psycopg2
from psycopg2 import sql
import os
from src.utils import get_sqlalchemy_engine

# Connexion à la base PostgreSQL
conn = get_sqlalchemy_engine()

conn.autocommit = True
cur = conn.cursor()

try:
    # Désactive temporairement les contraintes
    cur.execute("SET session_replication_role = 'replica';")

    # Récupère toutes les tables du schéma public
    cur.execute("""
        SELECT tablename FROM pg_tables
        WHERE schemaname = 'public';
    """)
    tables = cur.fetchall()

    # Vide chaque table
    for table in tables:
        table_name = table[0]
        print(f"🧹 TRUNCATE {table_name}...")
        cur.execute(sql.SQL("TRUNCATE TABLE {} RESTART IDENTITY CASCADE;").format(
            sql.Identifier(table_name)
        ))

    print("✅ Toutes les tables ont été vidées.")

finally:
    # Réactive les contraintes
    cur.execute("SET session_replication_role = 'origin';")
    cur.close()
    conn.close()
