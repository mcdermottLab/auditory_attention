#!/bin/bash
read -r num < swc_samp_rates.txt
if [[ "$num" -lt 30000 ]]; then
   echo "Take action!"
else
   echo "All OK!"
fi
