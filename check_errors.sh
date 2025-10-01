#!/bin/bash
# Quick script to check for recent 500 errors

echo "🔍 Checking for recent 500 errors..."
echo "=================================="

LOG_DIR="./logs"

if [ ! -d "$LOG_DIR" ]; then
    echo "❌ Logs directory not found at $LOG_DIR"
    echo "Logs will be created when errors occur."
    exit 1
fi

# Show recent errors from the last hour
echo "📋 Recent errors (last hour):"
find "$LOG_DIR" -name "*.log" -type f -exec grep -l "ERROR" {} \; | while read logfile; do
    echo ""
    echo "📁 From $logfile:"
    # Show errors from last hour (approximately)
    tail -100 "$logfile" | grep "ERROR" | tail -10
done

echo ""
echo "💡 For more detailed analysis, run:"
echo "   python manage.py monitor_errors --summary"
echo "   python manage.py monitor_errors --tail 20"
echo "   python manage.py monitor_errors --watch"