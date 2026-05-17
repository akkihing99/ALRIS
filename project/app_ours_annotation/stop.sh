#!/bin/bash

echo "Stopping all gunicorn annotation servers..."

pkill -f "gunicorn.*app_multi_gpu:app"

echo "All servers stopped."