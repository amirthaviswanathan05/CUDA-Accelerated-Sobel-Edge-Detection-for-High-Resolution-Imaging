#!/usr/bin/env bash
echo "Cleaning and Building Project..."
make clean
make build

echo "Starting Execution..."
./sobel_detector > artifacts/execution_log.txt

echo "Verification: Output file created in artifacts/ folder."
