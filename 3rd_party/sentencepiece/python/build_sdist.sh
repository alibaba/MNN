#!/bin/sh

mkdir -p sentencepiece

for i in CMakeLists.txt LICENSE README.md VERSION.txt cmake config.h.in  sentencepiece.pc.in src third_party
do
  echo "copying ../${i} sentencepiece/${i}"
  cp -f -R "../${i}" sentencepiece
done

python3 setup.py sdist
