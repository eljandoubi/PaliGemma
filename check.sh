#!/bin/bash

if [ -z "$HF_TOKEN" ]; then
  echo "HF_TOKEN is not set"
else
  VAR_VALUE="$HF_TOKEN"
  FIRST_4="${VAR_VALUE:0:4}"
  REMAINING="${VAR_VALUE:4}"
  MASKED=$(printf '%s' "$REMAINING" | sed 's/./*/g')
  echo "HF_TOKEN is set to ${FIRST_4}${MASKED}"
fi
