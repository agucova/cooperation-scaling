#!/bin/bash

while true; do
    python cooperation_scaling/main_noisy.py
    exit_status=$?
    if [ $exit_status -eq 0 ]; then
        echo "✅ Script exited normally"
        break
    else
        echo "⚠ Script terminated with status $exit_status. Restarting..."
        sleep 2
    fi
done
