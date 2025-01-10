#!/bin/bash

echo "<> this is to add the path to the tools (scripts) to the PATH variable; makes them executable from anywhere"
echo "<> the path must be the path of the file that you want to run as a command"  
echo "<> NOTE: THIS PATH WILL BE ADDED PERMANENTLY TO THE PATH VARIABLE ON ~/.bashrc, SO MAKE SURE YOU WANT TO DO THIS"
echo
echo "Please enter the ABSOLUTE path of the tool/script: (e for exit)"  

read path_to_tool

if [ $path_to_tool == "e" ]; then
    echo "exiting.."
    exit 0
fi

#checking if path exists
if [ -f $path_to_tool ]; then
    echo "<> file ${path_to_tool} exists: adding to PATH variable"
    # -- make sure the file is executable
    if [ ! -x $path_to_tool ]; then
        echo "<> file is not executable; making it executable"  
        chmod +x $path_to_tool  
    fi
elif [ -d $path_to_tool ]; then
    echo "<> directory ${path_to_tool} exists, but it is not a file; do you want to add it? (y/n)"
    read add_dir
    if [ $add_dir == "y" ]; then
        echo "<> adding directory to path"
    else
        echo "exiting.."
    fi
else
    echo "error -- path does not exist"
    exit 1
fi



# -- Add the path to the tools to the PATH variable
echo 'export PATH=$PATH:'${path_to_tool} >> ~/.bashrc
echo "<> PATH variable updated in ~/.bashrc"

source ~/.bashrc
echo "<> PATH variable updated in the current shell"
echo "<> p.s. you might need to run source ~/.bashrc if the tool does not run by calling it instantly"
 
