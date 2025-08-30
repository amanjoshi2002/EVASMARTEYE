import sqlite3

DB_PATH = "cameras.db"

def add_cameras():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Create table if not exists
    c.execute("""
        CREATE TABLE IF NOT EXISTS Camera (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            ip TEXT NOT NULL
        )
    """)
    # Insert cameras from 11 to 26
    for i in range(11, 27):
        name = f"Camera_{i}"
        ip = f"192.168.3.{i}"
        c.execute("INSERT INTO Camera (name, ip) VALUES (?, ?)", (name, ip))
    conn.commit()
    conn.close()
    print("Cameras added.")

if __name__ == "__main__":
    add_cameras()