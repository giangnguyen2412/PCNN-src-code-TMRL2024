#!/bin/sh

while getopts "d:s:o:" opt
do
  case "$opt" in
      d) paramA="$OPTARG" ;;
      s) paramB="$OPTARG" ;;
      o) paramC="$OPTARG" ;;
      ?)
  esac
done

if [ ! -d "$paramC" ]  # Check if the dataset folder exists?
  then
    mkdir "$paramC"
  fi

file_cnt=0

cd "$paramA" || exit

for d in */ ; do
  cd "$d"/ || exit
  pwd

  file_num=$(find . -type f | wc -l)
  file_cnt=$((file_cnt+file_num))

  if [ ! -d "$paramC"/"$d" ]  # Check if the wnid folder exists?
  then
    mkdir "$paramC"/"$d"
  fi
  find . -maxdepth 1 -type f | sort -R | head -"$file_num" | xargs cp -t "$paramC"/"$d"
  cd ..

  if [ $file_cnt -ge "$paramB" ]
  then
    exit 1
  fi
done
