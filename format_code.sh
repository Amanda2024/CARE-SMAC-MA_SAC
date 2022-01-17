#!/bin/bash

function read_dir(){
    for file in `ls $1`
    do
        if [ -d $1"/"$file ]
        then
            read_dir $1"/"$file
        else
            echo $1"/"$file
        fi
    done
}

read_dir ./ > file_list.txt
file_list=file_list.txt


for i in `cat $file_list`
do
	if [[ "$i" =~ py$ ]];then
		black	$i
	fi
done

