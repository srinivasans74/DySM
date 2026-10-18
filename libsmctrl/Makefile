# Set this if CUDA is installed in a different location
CUDA ?= /usr/local/cuda
# Note that CXX and CC are predefined as g++ and cc (respectively) by Make
NVCC ?= $(CUDA)/bin/nvcc
# Everything has to have -lcuda, as it's needed for libsmctrl
LDFLAGS := -lcuda -I$(CUDA)/include -L$(CUDA)/lib64

.PHONY: clean tests all

# ----- Main Library -----
libsmctrl.so: libsmctrl.c libsmctrl.h
	$(CC) $< -shared -o $@ -fPIC $(CFLAGS) $(LDFLAGS)

# -fPIC is needed even if built as a static library, in case we are linked into
# another shared library
libsmctrl.a: libsmctrl.c libsmctrl.h
	$(CC) $< -c -o libsmctrl.o -fPIC $(CFLAGS) $(LDFLAGS)
	ar rcs $@ libsmctrl.o

# ----- Utilities -----
# Use static linking with tests to avoid LD_LIBRARY_PATH issues
libsmctrl_test_gpc_info: libsmctrl_test_gpc_info.c libsmctrl.a testbench.h
	$(CC) $< -o $@ -g -L. -l:libsmctrl.a $(CFLAGS) $(LDFLAGS)

# ----- Tests -----
libsmctrl_test_mask_shared.o: libsmctrl_test_mask_shared.cu testbench.h
	$(NVCC) -ccbin $(CXX) $< -c -g

libsmctrl_test_global_mask: libsmctrl_test_global_mask.c libsmctrl.a libsmctrl_test_mask_shared.o
	$(NVCC) -ccbin $(CXX) $@.c -o $@ libsmctrl_test_mask_shared.o -g -L. -l:libsmctrl.a $(LDFLAGS)

libsmctrl_test_stream_mask: libsmctrl_test_stream_mask.c libsmctrl.a libsmctrl_test_mask_shared.o
	$(NVCC) -ccbin $(CXX) $@.c -o $@ libsmctrl_test_mask_shared.o -g -L. -l:libsmctrl.a $(LDFLAGS)

libsmctrl_test_stream_mask_override: libsmctrl_test_stream_mask_override.c libsmctrl.a libsmctrl_test_mask_shared.o
	$(NVCC) -ccbin $(CXX) $@.c -o $@ libsmctrl_test_mask_shared.o -g -L. -l:libsmctrl.a $(LDFLAGS)

libsmctrl_test_next_mask: libsmctrl_test_next_mask.c libsmctrl.a libsmctrl_test_mask_shared.o
	$(NVCC) -ccbin $(CXX) $@.c -o $@ libsmctrl_test_mask_shared.o -g -L. -l:libsmctrl.a $(LDFLAGS)

libsmctrl_test_next_mask_override: libsmctrl_test_next_mask_override.c libsmctrl.a libsmctrl_test_mask_shared.o
	$(NVCC) -ccbin $(CXX) $@.c -o $@ libsmctrl_test_mask_shared.o -g -L. -l:libsmctrl.a $(LDFLAGS)

tests: libsmctrl_test_gpc_info libsmctrl_test_global_mask libsmctrl_test_stream_mask libsmctrl_test_stream_mask_override libsmctrl_test_next_mask libsmctrl_test_next_mask_override

all: libsmctrl.so tests

clean:
	rm -f libsmctrl.so libsmctrl.a libsmctrl_test_gpu_info \
	      libsmctrl_test_mask_shared.o libmsctrl_test_global_mask \
	      libsmctrl_test_stream_mask libmsctrl_test_stream_mask_override \
	      libsmctrl_test_next_mask libmsctrl_test_next_mask_override
