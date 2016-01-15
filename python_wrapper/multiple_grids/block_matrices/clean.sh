#!/bin/bash

CURRENT_DIR_PATH=$(pwd)
EXECUTE_DIR_PATH="/home/federico/WorkSpace/PythonProjects/PhdThesis/PABLitO/python_wrapper/multiple_grids/block_matrices"

if [ "$EXECUTE_DIR_PATH" == "$CURRENT_DIR_PATH" ]; then
    SUB_DIR="./data"
    
    if [ -d "$SUB_DIR" ]; then
        cd $SUB_DIR

        count=`ls -1 ./*.vtu 2>/dev/null | wc -l`

        if [ $count != 0 ]; then 
            rm -v ./*.vtu
        fi

        count=`ls -1 ./*.pvtu 2>/dev/null | wc -l`

        if [ $count != 0 ]; then 
            rm -v ./*.pvtu
        fi 

        [ -f ./multiple_PABLO.vtm ] && rm -v ./multiple_PABLO.vtm
        
        cd ../
    fi

    SUB_DIR="./log"
    
    if [ -d "$SUB_DIR" ]; then
        cd $SUB_DIR

        count=`ls -1 *.log 2>/dev/null | wc -l`

        if [ $count != 0 ]; then 
            rm -v ./*.log
        fi

        cd ../
    fi

else
    echo "Execution path must be $EXECUTE_DIR_PATH"
fi
    
