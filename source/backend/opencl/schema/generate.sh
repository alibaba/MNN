#!/bin/bash

# check is flatbuffer installed or not
FLATC=../../../../../3rd_party/flatbuffers/tmp/flatc

# clean up
echo "*** cleaning up ***"
rm -f current/*.h
[ ! -d current ] && mkdir current

# flatc all fbs
pushd current > /dev/null
echo "*** generating fbs under $DIR ***"
find ../*.fbs | xargs ${FLATC} -c -b --gen-object-api --reflect-names
popd > /dev/null

# finish
echo "*** done ***"
