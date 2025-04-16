from sqlalchemy import create_engine

engine = create_engine("postgresql+psycopg2://user:password@localhost:5433/mspr_warehouse")

with engine.connect() as conn:
    result = conn.execute("SELECT 1")
    print(result.scalar())  # Doit afficher 1
