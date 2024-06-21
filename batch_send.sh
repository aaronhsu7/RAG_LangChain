#!/bin/bash
while read -r line; do python3 query.py "$line"; done < prompts.txt
