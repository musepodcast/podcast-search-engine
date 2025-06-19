# search_podcasts.py

import psycopg2
from psycopg2 import sql

def connect_db(dbname, user, password, host='localhost', port='5432'):
    """
    Establish a connection to the PostgreSQL database.
    """
    try:
        conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )
        print("Successfully connected to the database.")
        return conn
    except Exception as e:
        print(f"Database connection failed: {e}")
        raise

def search_podcasts(conn, search_phrase):
    """
    Search for podcasts containing the search_phrase in their transcript.

    Parameters:
    - conn: psycopg2 connection object.
    - search_phrase: str, the phrase to search for.

    Returns:
    - List of tuples containing episode details.
    """
    try:
        with conn.cursor() as cursor:
            # Convert the search phrase into tsquery format for phrase matching
            # For exact phrase matching, use <-> operator between words
            # Example: "phones with 6g technology" becomes "phones <-> with <-> 6g <-> technology"
            ts_query = ' <-> '.join(search_phrase.strip().split())

            query = '''
                SELECT
                    Channels.channel_title,
                    Episodes.episode_title,
                    Episodes.publication_date,
                    Episodes.description,
                    Episodes.transcript,
                    ts_rank(Episodes.tsv_transcript, to_tsquery('english', %s)) AS rank
                FROM
                    Episodes
                JOIN
                    Channels ON Episodes.channel_id = Channels.id
                WHERE
                    Episodes.tsv_transcript @@ to_tsquery('english', %s)
                ORDER BY
                    rank DESC;
            '''
            cursor.execute(query, (ts_query, ts_query))
            results = cursor.fetchall()
            return results
    except Exception as e:
        print(f"Search query failed: {e}")
        return []

if __name__ == "__main__":
    # Database connection parameters
    DB_NAME = "podcast_db"
    DB_USER = "postgres"     # Replace with your PostgreSQL username
    DB_PASSWORD = "root" # Replace with your PostgreSQL password
    DB_HOST = "localhost"
    DB_PORT = "5432"

    # Connect to the database
    connection = connect_db(DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT)

    # Example search phrase
    search_input = "sexy and saying"  

    # Perform the search
    episodes = search_podcasts(connection, search_input)

    # Display results
    if episodes:
        for ep in episodes:
            channel, title, pub_date, description, transcript, rank = ep
            print(f"Channel: {channel}")
            print(f"Title: {title}")
            print(f"Published on: {pub_date}")
            print(f"Description: {description}")
            print(f"Transcript Snippet: {transcript[:150]}...")
            print(f"Relevance Score: {rank}")
            print("-" * 50)
    else:
        print(f"No matching episodes found for \"{search_input}\".")

    # Close the database connection
    connection.close()
