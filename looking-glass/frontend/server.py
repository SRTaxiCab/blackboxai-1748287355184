from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.dirname(os.path.abspath(__file__)), **kwargs)

    def log_message(self, format, *args):
        logger.info(format % args)

def run_server(port=8000):
    """Run the HTTP server"""
    try:
        server_address = ('', port)
        httpd = HTTPServer(server_address, CustomHandler)
        logger.info(f'Starting server on port {port}...')
        logger.info(f'Open http://localhost:{port} in your browser')
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info('Server stopped by user')
        httpd.server_close()
    except Exception as e:
        logger.error(f'Error running server: {str(e)}')
        raise

if __name__ == '__main__':
    run_server()
