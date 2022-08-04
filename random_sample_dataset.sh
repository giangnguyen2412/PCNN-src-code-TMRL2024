#!/bin/sh
# sample usage:
# sh random_sample_dataset.sh -d /home/giang/Downloads/datasets/imagenet1k-val -s 5 -o /home/giang/Downloads/datasets/imagenet5k-1k

while getopts "d:s:o:" opt
do
  case "$opt" in
      d) paramA="$OPTARG" ;;
      s) paramB="$OPTARG" ;;
      o) paramC="$OPTARG" ;;
      ?)
  esac
done

rm -rf "$paramC"
mkdir "$paramC"


cd "$paramA" || exit

for d in */ ; do
  cd "$d"/ || exit
  pwd
  # Remove the wnid folders
  rm -rf "$paramC"/"$d"
  mkdir "$paramC"/"$d"
  find . -maxdepth 1 -type f | sort -R | head -"$paramB" | xargs cp -t "$paramC"/"$d"
  cd ..
done
