# debugging.py
import logging
import traceback

def setup_logging():
    logging.basicConfig(
        filename='app_debug.log',
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
def log_exception(exc: Exception):
    logging.error(f"CRITICAL ERROR: {str(exc)}")
    logging.error(traceback.format_exc())
    
    # Envoi d'alerte (optionnel)
    try:
        import requests
        requests.post('https://api.yourapp.com/errors', json={
            'error': str(exc),
            'traceback': traceback.format_exc()
        })
    except:
        pass
