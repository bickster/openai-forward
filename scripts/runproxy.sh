#!/bin/bash

echo "Starting Proxy"
cd /home/ec2-user/openai-forward
python3 -m openai_forward run --port=9999 --workers=1 --log_chat=true

