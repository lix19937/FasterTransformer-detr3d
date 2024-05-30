
cd build
rm -fr ../build/c*  ../build/C*

# cmake -DSM=86 -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_TRT=ON .. && make  -j44

# 

cmake  -DSM=80 -DNDEBUG=ON  -DBUILD_PTQ_PHASE=ON  -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=OFF -DBUILD_TRT=ON .. && make  -j44
