import sqlite3
import json
from datetime import datetime, timedelta # Added timedelta for example

DATABASE_NAME = "metrics.db"

def init_db():
    """Initializes the database and creates the metrics table if it doesn't exist."""
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                name TEXT NOT NULL,
                value REAL NOT NULL,
                labels TEXT
            )
        ''')
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error initializing database: {e}")
    finally:
        if conn:
            conn.close()

def insert_metric(timestamp: str, name: str, value: float, labels: dict):
    """Inserts a new metric into the database."""
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        labels_json = json.dumps(labels) if labels else None
        cursor.execute('''
            INSERT INTO metrics (timestamp, name, value, labels)
            VALUES (?, ?, ?, ?)
        ''', (timestamp, name, value, labels_json))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error inserting metric: {e}")
    finally:
        if conn:
            conn.close()

def get_aggregated_metrics(
    metric_name: str,
    start_time: datetime,
    end_time: datetime,
    aggregation_period: str, # e.g., "hourly", "daily"
    aggregation_function: str # e.g., "AVG", "SUM"
) -> list:
    """
    Queries aggregated historical metrics from the database.
    """
    results = []
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()

        time_format_sql = ""
        if aggregation_period == "hourly":
            time_format_sql = "strftime('%Y-%m-%d %H:00:00', timestamp)"
        elif aggregation_period == "daily":
            time_format_sql = "strftime('%Y-%m-%d 00:00:00', timestamp)"
        # Optional: Add weekly aggregation
        # elif aggregation_period == "weekly":
        #     time_format_sql = "strftime('%Y-%W 00:00:00', timestamp)" # Week starts on Monday
        else:
            raise ValueError("Invalid aggregation_period. Supported values: 'hourly', 'daily'.")

        if not aggregation_function.upper() in ["AVG", "SUM", "COUNT", "MIN", "MAX"]:
            raise ValueError("Invalid aggregation_function. Supported functions: AVG, SUM, COUNT, MIN, MAX.")

        # Construct the SQL query
        # Convert datetime objects to ISO 8601 string format for SQLite
        start_time_str = start_time.isoformat()
        end_time_str = end_time.isoformat()

        query = f"""
            SELECT
                {time_format_sql} as time_bucket,
                {aggregation_function.upper()}(value) as aggregated_value
            FROM
                metrics
            WHERE
                name = ? AND
                timestamp >= ? AND
                timestamp <= ?
            GROUP BY
                time_bucket
            ORDER BY
                time_bucket;
        """

        cursor.execute(query, (metric_name, start_time_str, end_time_str))
        rows = cursor.fetchall()

        for row in rows:
            results.append({'time_bucket': row[0], 'aggregated_value': row[1]})

    except sqlite3.Error as e:
        print(f"Database error in get_aggregated_metrics: {e}")
    except ValueError as ve:
        print(f"Value error in get_aggregated_metrics: {ve}")
    except Exception as e:
        print(f"Unexpected error in get_aggregated_metrics: {e}")
    finally:
        if conn:
            conn.close()
    return results

if __name__ == '__main__':
    init_db()
    print(f"Database '{DATABASE_NAME}' initialized and 'metrics' table created (if it didn't exist).")

    # Example dummy data insertion
    now = datetime.utcnow()
    # Clean up old test data for consistent results if script is run multiple times
    try:
        conn_main = sqlite3.connect(DATABASE_NAME)
        cursor_main = conn_main.cursor()
        cursor_main.execute("DELETE FROM metrics WHERE name LIKE 'test_metric_%'")
        conn_main.commit()
    except sqlite3.Error as e:
        print(f"Error cleaning up old test data: {e}")
    finally:
        if conn_main:
            conn_main.close()


    print("\nInserting dummy metrics for testing get_aggregated_metrics...")
    insert_metric((now - timedelta(hours=2, minutes=30)).isoformat(), "test_metric_avg", 10, {"source":"test"})
    insert_metric((now - timedelta(hours=1, minutes=30)).isoformat(), "test_metric_avg", 20, {"source":"test"})
    insert_metric((now - timedelta(hours=0, minutes=30)).isoformat(), "test_metric_avg", 30, {"source":"test"}) # Current hour
    insert_metric((now - timedelta(days=1, hours=1)).isoformat(), "test_metric_avg", 5, {"source":"test"}) # Yesterday

    insert_metric((now - timedelta(hours=1)).isoformat(), "test_metric_sum", 100, {"source":"test"})
    insert_metric((now - timedelta(hours=1)).isoformat(), "test_metric_sum", 50, {"source":"test"})
    insert_metric((now - timedelta(days=1, hours=2)).isoformat(), "test_metric_sum", 200, {"source":"test"}) # Yesterday
    print("Dummy metrics inserted.")

    # Example query
    print("\nQuerying aggregated metrics...")
    hourly_avg_data = get_aggregated_metrics(
        metric_name="test_metric_avg",
        start_time=now - timedelta(days=2), # Look back 2 days
        end_time=now,
        aggregation_period="hourly",
        aggregation_function="AVG"
    )
    print("\nHourly AVG for 'test_metric_avg':")
    for row in hourly_avg_data:
        print(row)

    daily_sum_data = get_aggregated_metrics(
        metric_name="test_metric_sum",
        start_time=now - timedelta(days=2), # Look back 2 days
        end_time=now,
        aggregation_period="daily",
        aggregation_function="SUM"
    )
    print("\nDaily SUM for 'test_metric_sum':")
    for row in daily_sum_data:
        print(row)
    
    hourly_sum_data = get_aggregated_metrics(
        metric_name="test_metric_sum",
        start_time=now - timedelta(days=2),
        end_time=now,
        aggregation_period="hourly",
        aggregation_function="SUM"
    )
    print("\nHourly SUM for 'test_metric_sum':")
    for row in hourly_sum_data:
        print(row)

    # Test case for no data
    no_data = get_aggregated_metrics(
        metric_name="non_existent_metric",
        start_time=now - timedelta(days=1),
        end_time=now,
        aggregation_period="hourly",
        aggregation_function="AVG"
    )
    print("\nHourly AVG for 'non_existent_metric' (should be empty):")
    print(no_data)

    # Test invalid aggregation period
    print("\nTesting invalid aggregation period:")
    invalid_period_data = get_aggregated_metrics("test_metric_avg", now - timedelta(days=1), now, "minutely", "AVG")
    print(invalid_period_data)

    # Test invalid aggregation function
    print("\nTesting invalid aggregation function:")
    invalid_function_data = get_aggregated_metrics("test_metric_avg", now - timedelta(days=1), now, "hourly", "MEDIAN")
    print(invalid_function_data)
