#!/bin/bash

# -- a script to export PYTHONPATH w user given path
#       if no given path then export current directory
#       need to rerun everytime a terminal session is started 
# Usage:
#       set-pythonpath.sh                       # --> adds current directory to PYTHONPATH
#       set-pythonpath.sh [path]                # --> adds given path to PYTHONPATH


PATH_TO_EXPORT=$1

if [ -z "$1" ]; then
    PATH_TO_EXPORT=$(pwd)
    echo ">>> No path is given, exporting current directory"
else
    PATH_TO_EXPORT=$(realpath "$1")
fi

# -- check if it's a valid python package
if [ ! -f "$PATH_TO_EXPORT/__init__.py" ]; then
    echo "<<< '$PATH_TO_EXPORT' is not a valid python package, make sure __init__.py exists"
    exit 1
fi

# -- check if the path is already in PYTHONPATH
if [[ ":$PYTHONPATH:" != *":$PATH_TO_EXPORT:"* ]]; then
    export PYTHONPATH="$PATH_TO_EXPORT:$PYTHONPATH"
    echo ">>> PYTHONPATH is exported with '$PATH_TO_EXPORT'"
else
    echo ">>> '$PATH_TO_EXPORT' is already in PYTHONPATH"
fi

# export PYTHONPATH=$PATH_TO_EXPORT:$PYTHONPATH 
# echo ">>> PYTHONPATH is exported with $PATH_TO_EXPORT"
 
echo " <> Current PYTHONPATH: $PYTHONPATH <>"