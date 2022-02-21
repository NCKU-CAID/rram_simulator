
g++ -o trans_weight trans_weight.cpp
./trans_weight

g++ -o genfloorplan genfloorplan.cpp
./genfloorplan

g++ -o genpower genpower.cpp
./genpower

../hotspot/Hotspot/hotspot -c ../hotspot.config -f test.flp -p test.ptrace -model_type grid -grid_steady_file test.grid.steady

perl ../hotspot/Hotspot/grid_thermal_map.pl test.flp test.grid.steady > test.svg


g++ -o genheatmap genheatmap.cpp
./genheatmap

