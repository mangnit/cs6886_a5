# CS6886 -- Optimising attention on a CPU
#
# Path to your C compiler. Use the LATEST clang, for example:
#   Linux:  /usr/bin/clang-23        macOS (Homebrew):  /opt/homebrew/opt/llvm/bin/clang
CC = /mnt/d/cs6886_a5_extra/tools/env/bin/clang

CFLAGS = -O3 -march=native
OMP    = -fopenmp
INPUTS = $(wildcard input/*.bin)

# make <variant>  : compile src/attn_<variant>.c and report its time on every input
# make all        : the same for every variant
# make check      : check every variant's output against naive's output
# make clean      : delete build/ and output/

all: naive loops tiled fused best omp

naive loops tiled fused best:
	@mkdir -p build output; rm -f build/$@ output/$@_*
	-$(CC) $(CFLAGS) -o build/$@ src/attn_$@.c -lm
	-@for f in $(INPUTS); do [ -x build/$@ ] && ./build/$@ $$f output/$@_`basename $$f`; done; true

omp:
	@mkdir -p build output; rm -f build/$@ output/$@_*
	-$(CC) $(CFLAGS) $(OMP) -o build/$@ src/attn_$@.c -lm
	-@for f in $(INPUTS); do [ -x build/$@ ] && ./build/$@ $$f output/$@_`basename $$f`; done; true

# Passes if max|O - O_naive| / max|O_naive| <= 1e-4. Outputs never match exactly:
# different loop orders round differently.
check:
	@python3 -c "$$CHECK"

clean:
	rm -rf build output

.PHONY: all naive loops tiled fused best omp check clean

define CHECK
import glob, os, struct
from array import array
def load(p):
    f = open(p, "rb"); n, d = struct.unpack("<ii", f.read(8)); a = array("f"); a.fromfile(f, n * d); return a
variants = ["loops", "tiled", "fused", "best", "omp"]
refs = sorted(glob.glob("output/naive_*.bin"))
if not refs: print("no naive output: run make naive first")
for ref_path in refs:
    ref = load(ref_path); scale = max(abs(x) for x in ref)
    inp = os.path.basename(ref_path)[len("naive_"):]
    for v in variants:
        path = "output/" + v + "_" + inp
        if not os.path.exists(path): continue
        out = load(path)
        err = max(abs(a - b) for a, b in zip(out, ref)) / scale if len(out) == len(ref) else float("inf")
        print("%-20s %-12s %s  (error %.1e)" % (inp, v, "PASS" if err <= 1e-4 else "FAIL", err))
missing = [v for v in variants if not glob.glob("output/" + v + "_*.bin")]
if missing: print("no output (not implemented, did not compile, or crashed): " + " ".join(missing))
endef
export CHECK
