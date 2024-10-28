#!/bin/bash

NSYS_PATH="nsys"
REPORT_PATH="/home/pp24/pp24s104/NTHU-ParallelProgramming/hw1/nsys_reports"

for i in {0..11}; do
    INPUT_FILE="${REPORT_PATH}/rank_${i}.nsys-rep"
    OUTPUT_FILE="rank_${i}.csv"
    
    ${NSYS_PATH} stats -r mpi_event_sum --format csv ${INPUT_FILE} > ${OUTPUT_FILE}
    
    if [ $? -eq 0 ]; then
        echo "Successfully generated ${OUTPUT_FILE}"
    else
        echo "Failed to generate ${OUTPUT_FILE} for rank_${i}"
    fi
done
