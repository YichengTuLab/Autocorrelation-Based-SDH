# Autocorrelation-Based SDH (ACS) -- build
#
# Requirements: a C compiler with OpenMP, and FFTW3 (single + OpenMP).
#   Debian/Ubuntu:  sudo apt-get install build-essential libfftw3-dev
#
# Usage:
#   make            # build ./acs
#   make clean

CC      ?= gcc
CFLAGS  ?= -O3 -fopenmp          # matches the paper's build; add -march=native for extra speed
LDLIBS  := -lfftw3 -lfftw3_omp -lm

BIN := acs
SRC := src/acs.c

$(BIN): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(BIN) $(LDLIBS)

clean:
	rm -f $(BIN)

.PHONY: clean
