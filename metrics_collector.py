import requests
from prometheus_client.parser import text_string_to_metric_families
import time
from datetime import datetime
import metrics_db
import logging
import os

# Configuration
METRICS_URL = os.environ.get("METRICS_ENDPOINT_URL", "http://localhost:8000/metrics")
SCRAPE_INTERVAL_SECONDS = int(os.environ.get("SCRAPE_INTERVAL", 10))

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_prometheus_metrics(text_data: str) -> list:
    """
    Parses Prometheus text format data and extracts metrics.
    Ignores '_bucket' metrics for simplicity.
    """
    parsed_metrics = []
    try:
        for family in text_string_to_metric_families(text_data):
            for sample in family.samples:
                name = sample.name
                value = sample.value
                labels = sample.labels

                # Skip histogram bucket metrics for now
                if name.endswith("_bucket"):
                    continue

                parsed_metrics.append({'name': name, 'value': value, 'labels': labels})
        logging.info(f"Successfully parsed {len(parsed_metrics)} metrics.")
    except Exception as e:
        logging.error(f"Error parsing Prometheus metrics: {e}")
    return parsed_metrics

def fetch_and_store_metrics():
    """
    Fetches metrics from the METRICS_URL, parses them, and stores them in the database.
    """
    metrics_db.init_db() # Ensure DB is initialized

    try:
        logging.info(f"Fetching metrics from {METRICS_URL}...")
        response = requests.get(METRICS_URL, timeout=5) # Added timeout
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
        logging.info("Successfully fetched metrics.")

        metrics_to_store = parse_prometheus_metrics(response.text)
        
        if not metrics_to_store:
            logging.info("No metrics to store after parsing.")
            return

        current_utc_timestamp = datetime.utcnow()
        inserted_count = 0
        for metric in metrics_to_store:
            try:
                metrics_db.insert_metric(
                    timestamp=current_utc_timestamp.isoformat(),
                    name=metric['name'],
                    value=metric['value'],
                    labels=metric['labels']
                )
                inserted_count += 1
            except Exception as e:
                logging.error(f"Error inserting metric '{metric['name']}' into database: {e}")
        
        if inserted_count > 0:
            logging.info(f"Successfully inserted {inserted_count} metrics into the database.")

    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching metrics from {METRICS_URL}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred in fetch_and_store_metrics: {e}")

if __name__ == '__main__':
    metrics_db.init_db() # Initial call to ensure DB is ready
    logging.info(f"Metrics collector started. Scraping {METRICS_URL} every {SCRAPE_INTERVAL_SECONDS} seconds.")
    
    try:
        while True:
            fetch_and_store_metrics()
            logging.info(f"Waiting for {SCRAPE_INTERVAL_SECONDS} seconds before next scrape...")
            time.sleep(SCRAPE_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        logging.info("Metrics collector shutting down gracefully.")
    except Exception as e:
        logging.error(f"Critical error in main loop: {e}", exc_info=True)
    finally:
        logging.info("Metrics collector stopped.")
