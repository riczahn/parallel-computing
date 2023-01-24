COMPILER:=nvcc
CFLAGS+=
LDLIBS+=-lm

SRC_FILES:=src/shallow_water.cu src/argument_utils.cpp

INC_PATHS:=-Iinc/ -I/usr/local/cuda/include

.PHONY: clean purge setup run check plot show viz

compile: $(SRC_FILES)
	$(COMPILER) $^ $(CFLAGS) $(LDLIBS) -o solution

clean:
	-rm -f solution

purge:
	-rm -f solution data/*.bin plots/*.png video/*.mp4

setup:
	-mkdir -p data plots video

run: purge compile
	./solution

check: purge compile
	gcc ./check/compare_solutions.c -lm -o ./check/compare_solutions
	./check/check_solution.sh

plot: purge run
	./plot_solution.sh

show: purge run viz

viz:
	./plot_solution.sh > /dev/null
	ffmpeg -framerate 10 -i plots/%05d.png video/output.mp4 &> /dev/null
	./open_video.sh &> /dev/null
