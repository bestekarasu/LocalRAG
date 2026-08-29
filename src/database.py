import sqlite3
from pathlib import Path
import json


DB_PATH = Path("db/localrag.db")


def create_database():
    """
    LocalRAG SQLite veritabanını oluşturur.
    """

    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            chunk TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
        """
    )

    connection.commit()
    connection.close()


def insert_chunk(source, chunk, embedding):
    """
    Bir chunk ve embedding'i SQLite'a kaydeder.
    """

    create_database()

    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO chunks (
            source,
            chunk,
            embedding
        )
        VALUES (?, ?, ?)
        """,
        (
            source,
            chunk,
            json.dumps(embedding)
        )
    )

    connection.commit()
    connection.close()


def delete_source(source):
    """
    Belirli bir belgeye ait eski chunkları siler.

    Böylece aynı belge tekrar indexlendiğinde
    veritabanında duplicate kayıt oluşmaz.
    """

    create_database()

    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    cursor.execute(
        """
        DELETE FROM chunks
        WHERE source = ?
        """,
        (source,)
    )

    deleted_count = cursor.rowcount

    connection.commit()
    connection.close()

    return deleted_count


def get_all_chunks():
    """
    Veritabanındaki tüm chunkları getirir.
    """

    create_database()

    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT id, source, chunk, embedding
        FROM chunks
        """
    )

    rows = cursor.fetchall()

    connection.close()

    return rows


def get_chunks_by_source(source):
    """
    Belirli bir belgeye ait chunkları getirir.
    """

    create_database()

    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT id, source, chunk, embedding
        FROM chunks
        WHERE source = ?
        """,
        (source,)
    )

    rows = cursor.fetchall()

    connection.close()

    return rows


def get_source_names():
    """
    Veritabanında indexlenmiş benzersiz belge
    isimlerini getirir.
    """

    create_database()

    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT DISTINCT source
        FROM chunks
        ORDER BY source
        """
    )

    rows = cursor.fetchall()

    connection.close()

    return [row[0] for row in rows]


def get_chunk_count():
    """
    Veritabanındaki toplam chunk sayısını getirir.
    """

    create_database()

    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM chunks
        """
    )

    count = cursor.fetchone()[0]

    connection.close()

    return count


if __name__ == "__main__":

    create_database()

    print("SQLite veritabanı hazır.")
    print(f"Konum: {DB_PATH}")

    print(
        f"Toplam chunk sayısı: "
        f"{get_chunk_count()}"
    )

    print(
        "Indexlenmiş belgeler:"
    )

    for source in get_source_names():
        print(f"- {source}")