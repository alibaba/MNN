root_dir=$1

if [ $# -eq 1 ]; then
    gcc -c -O3 kernel.c -o kernel.o
else
    clang -c -O3 kernel.bc -o kernel.o
fi

g++ -I$root_dir/include -I$root_dir/schema/current -I$root_dir/3rd_party/flatbuffers/include -I./ -std=c++11 -fPIC -O3 -o ./PluginWrapper.cpp.o -c $root_dir/codegen/plugin_wrapper/PluginWrapper.cpp -o ./Plugin.cpp.o
g++ -fPIC -std=c++11 -O3 -shared -o libplugin_fuse.so ./Plugin.cpp.o ./kernel.o ./libMNN_Plugin.dylib
