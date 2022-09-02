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


cd "$paramA" || exit

for d in */ ; do
  cd "$d"/ || exit
  pwd

  if [ ! -d "$paramC"/"$d" ]  # Check if the wnid folder exists?
  then
    mkdir "$paramC"/"$d"
  fi
  find . -maxdepth 1 -type f | sort -R | head -"$paramB" | xargs cp -t "$paramC"/"$d"
  cd ..
done
