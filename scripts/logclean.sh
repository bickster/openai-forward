#!/bin/bash

/usr/bin/find /home/ec2-user/openai-forward/Log/ -mtime +7 -name "*.log" -print -delete
