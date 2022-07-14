#!/bin/sh

rm -rf /home/giang/Downloads/datasets/random_sample_dataset/
mkdir /home/giang/Downloads/datasets/random_sample_dataset/

while getopts "d:s:" opt
do
  case "$opt" in
      d) paramA="$OPTARG" ;;
      s) paramB="$OPTARG" ;;
      ?)
  esac
done

cd "$paramA" || exit

for d in */ ; do
  cd "$d"/ || exit
  pwd
  # Remove the wnid folders
  rm -rf /home/giang/Downloads/datasets/random_sample_dataset/"$d"
  mkdir /home/giang/Downloads/datasets/random_sample_dataset/"$d"
  find . -maxdepth 1 -type f | sort -R | head -"$paramB" | xargs cp -t /home/giang/Downloads/datasets/random_sample_dataset/"$d"
  cd ..
done