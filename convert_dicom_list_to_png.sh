#!/bin/bash
filelist=$1
OUT_DIRECTORY="/data/UCSF_MAMMO/2017-07-png-16bit"
TMPFILE="tmp.dcm"

nn=0
for file in $filelist; do
    nn=$((nn+1))
    dirn_=$(dirname $file)
    id=`echo $file | awk -v FS="/" '{for (nn=NF-3;nn<NF;nn++){ printf $nn "_" }; print $NF }'`
    id=${id%.dcm}
    #echo  $id; 
    # gdcmconv --raw "$file" $TMPFILE
    # convert $TMPFILE -resize 299x299! $OUT_DIRECTORY/$id.png
    gdcmconv --raw "$file" /dev/stdout | convert /dev/stdin -depth 16 -resize 299x299! $OUT_DIRECTORY/$id.png
    #-depth
    if [ $((nn % 40)) -eq 0 ]; then
    echo "$nn"
fi
done
