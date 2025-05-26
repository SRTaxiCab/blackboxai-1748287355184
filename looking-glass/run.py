import uvicorn
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def setup_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        "data/raw",
        "data/processed",
        "models/saved",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import fastapi
        import numpy
        import torch
        import transformers
        import sentence_transformers
        import aiohttp
        logger.info("All required dependencies are installed")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {str(e)}")
        return False

async def initialize_services():
    """Initialize all required services"""
    try:
        # Import services
        from backend.scraper.data_collector import DataCollector
        from backend.utils.analyzer import SignalAnalyzer, TrendAnalyzer
        from backend.models.timeline import TimelineProjector

        # Initialize services
        data_collector = DataCollector()
        signal_analyzer = SignalAnalyzer()
        trend_analyzer = TrendAnalyzer()
        timeline_projector = TimelineProjector()

        logger.info("All services initialized successfully")
        return {
            "data_collector": data_collector,
            "signal_analyzer": signal_analyzer,
            "trend_analyzer": trend_analyzer,
            "timeline_projector": timeline_projector
        }
    except Exception as e:
        logger.error(f"Error initializing services: {str(e)}")
        raise

def run_api():
    """Run the FastAPI server"""
    try:
        import uvicorn
        from backend.api.main import app
        
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", 8000))
        
        logger.info(f"Starting API server on {host}:{port}")
        uvicorn.run(app, host=host, port=port)
    except Exception as e:
        logger.error(f"Error starting API server: {str(e)}")
        raise

async def run_system_check():
    """Perform a system check by running a test analysis"""
    try:
        services = await initialize_services()
        
        # Test data collection
        test_data = await services["data_collector"].scrape_social_media(
            platform="twitter",
            query="artificial intelligence",
            start_date=datetime.now() - timedelta(days=1),
            end_date=datetime.now()
        )
        logger.info(f"Test data collection successful: {len(test_data)} items collected")

        # Test analysis
        if test_data:
            analysis = services["trend_analyzer"].analyze_trends(test_data)
            logger.info("Test analysis successful")

        # Test timeline projection
        timeline = services["timeline_projector"].project_timeline([])
        logger.info("Test timeline projection successful")

        return True
    except Exception as e:
        logger.error(f"System check failed: {str(e)}")
        return False

async def main():
    """Main entry point"""
    try:
        # Setup directories
        setup_directories()

        # Check dependencies
        if not check_dependencies():
            logger.error("Dependency check failed")
            return

        # Run system check
        system_check_result = await run_system_check()
        if not system_check_result:
            logger.error("System check failed")
            return

        # Start API server
        run_api()

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    # Run the main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down Looking Glass AI")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise
