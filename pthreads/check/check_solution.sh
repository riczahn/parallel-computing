echo
echo "Comparing solution with reference data"
echo "-----------------------------------------------------------"
echo "Grid size: 256, Max iteration: 5000, Snapshot frequency: 40"
echo
./solution 1>/dev/null
./check/compare_solutions 256 data/00120.bin check/references/n256/00120.bin
echo
