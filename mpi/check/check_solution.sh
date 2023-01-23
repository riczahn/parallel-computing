echo
echo "Comparing solution with reference data"
echo "------------------------------------------------------------"
echo "Grid size: 1024, Max iteration: 100000, Snapshot frequency: 1000"
echo
echo "Checking evenly divisable grid size"
echo "------------------------------------------------------------------"
echo
for i in 1 2 4 8
do
    echo "Running with $i processes:"
    mpirun -n $i --oversubscribe ./solution 1>/dev/null
    ./check/compare_solutions 256 data/00050.bin check/references/n256/00050.bin
    echo
done
