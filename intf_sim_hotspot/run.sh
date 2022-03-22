#cd ../hotspot/HotSpot
#make clean
#make

#cd ../../intf_sim_hotspot

./weight2power ./weight_file.txt weight2power.cfg weight_sum.txt

./genfloorplan

./genpower circuit_power.cfg weight.cfg power_consumption.ptrace

../hotspot/HotSpot/hotspot -c ../hotspot/HotSpot/hotspot.config -f test.flp -p power_consumption.ptrace -model_type grid -grid_steady_file test.grid.steady

perl ../hotspot/HotSpot/grid_thermal_map.pl test.flp test.grid.steady > output.svg

./genheatmap ./heatmap.config ./test.grid.steady ./heatmap.txt

echo "Output file heatmap.txt"
