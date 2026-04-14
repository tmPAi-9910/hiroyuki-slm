#!/usr/bin/env python
"""
Hiroyuki-SLM API Implementation
This module implements the API endpoints for the Hiroyuki-SLM model using FastAPI.
"""
import logging
import api

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting Hiroyuki-SLM API server...")
    api.start()
