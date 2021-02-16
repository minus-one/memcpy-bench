#! /bin/bash

for((x=4096; x<=1073741824; x=$x*2))
do
  #./test --memSize=1073741824 --chunkSize=$x --dtoh >> dtoh_chunked_memcpy_time.out
  ./test --memSize=$x --dtoh >> dtoh_async_memcpy_time.out
done
