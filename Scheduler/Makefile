# ============================================================
# Host architecture check (x86_64 only)
# ============================================================
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),x86_64)
  $(info [INFO] Building for x86_64 host: $(UNAME_M))
else
  $(error This Makefile is configured for x86_64, but host is $(UNAME_M))
endif

# ============================================================
# Compiler Settings
# ============================================================
CXX       = g++
NVCC      = $(CUDA_PATH)/bin/nvcc
PROJECT_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
# Base flags for CPU and CUDA (x86_64)
CXXFLAGS  = -std=c++17 -pthread -O0 -g -Wall -m64 -fno-omit-frame-pointer
CXXFLAGS += -I$(PROJECT_ROOT)/json/single_include
NVCCFLAGS = -std=c++17 -O0 -G -g -m64 \
            --default-stream per-thread \
            --extended-lambda --expt-relaxed-constexpr \
            -Xcompiler "-fno-omit-frame-pointer" \
            -arch=sm_120

# ============================================================
# CUDA Path Auto-Detection
# ============================================================
CUDA_PATH     ?= /usr/local/cuda
CUDA_LIB_PATH := $(CUDA_PATH)/lib64
CUDA_INC_PATH := $(CUDA_PATH)/include

# ============================================================
# Directories
# ============================================================
SRC_DIR           = .
LIBSMCTRL_DIR     = ../libsmctrl

# Application directories
APP_DIR        = ../application
HIST_DIR       = $(APP_DIR)/hist
HOTSPOT_DIR    = $(APP_DIR)/hotspot
PROJECTION_DIR = $(APP_DIR)/projection
STENCIL_DIR    = $(APP_DIR)/stencil
STEREODISP_DIR = $(APP_DIR)/stereodisparity
VECTORADD_DIR  = $(APP_DIR)/vectoradd
SRAD_DIR       = $(APP_DIR)/srad
MATTINT_DIR    = $(APP_DIR)/matrixmulinter
MTTKRP_DIR    = $(APP_DIR)/mttkrp

# ============================================================
# Source Files
# ============================================================
CPP_SRCS = $(SRC_DIR)/scheduler.cpp \
           $(SRC_DIR)/controller.cpp \
           $(SRC_DIR)/parameters.cpp \
           $(SRC_DIR)/cuda_runner.cpp \
           $(SRC_DIR)/stgm.cpp \


CUDA_SRCS = \
    $(wildcard $(HIST_DIR)/*.cu) \
    $(wildcard $(HOTSPOT_DIR)/*.cu) \
    $(wildcard $(PROJECTION_DIR)/*.cu) \
    $(wildcard $(STENCIL_DIR)/*.cu) \
    $(wildcard $(STEREODISP_DIR)/*.cu) \
    $(wildcard $(VECTORADD_DIR)/*.cu) \
    $(wildcard $(SRAD_DIR)/*.cu) \
    $(wildcard $(MTTKRP_DIR)/*.cu) \
    $(wildcard $(MATTINT_DIR)/*.cu)

OBJS_CPP  = $(CPP_SRCS:.cpp=.o)
OBJS_CUDA = $(CUDA_SRCS:.cu=.o)
OBJS      = $(OBJS_CPP) $(OBJS_CUDA)

TARGET = scheduler

# ============================================================
# Build Rules
# ============================================================
all: $(TARGET)

# Rebuild when gain.py exports new controller parameters.
controller.o: controller.h gain_matrix.h
scheduler.o: controller.h gain_matrix.h

$(TARGET): $(OBJS)
	@echo "[LINK] Building final x86_64 binary..."
	$(CXX) -g -O0 -m64 -o $@ $^ \
	    -I$(CUDA_INC_PATH) \
	    -I$(LIBSMCTRL_DIR) \
	    -L$(LIBSMCTRL_DIR) -L$(CUDA_LIB_PATH) \
	    -Wl,-rpath,'$$ORIGIN/../libsmctrl' -Wl,-rpath,$(CUDA_LIB_PATH) \
	    -lsmctrl -lcudart -lcuda -lpthread

# ------------------------------------------------------------
# CPU Compilation
# ------------------------------------------------------------
%.o: %.cpp
	@echo "[CXX] Compiling $< ..."
	$(CXX) $(CXXFLAGS) -I$(LIBSMCTRL_DIR) -I$(CUDA_INC_PATH) -c $< -o $@

# ------------------------------------------------------------
# CUDA Compilation
# ------------------------------------------------------------
%.o: %.cu
	@echo "[NVCC] Compiling $< ..."
	$(NVCC) $(NVCCFLAGS) -I$(LIBSMCTRL_DIR) -I$(CUDA_INC_PATH) -c $< -o $@

# ============================================================
# Utility Targets
# ============================================================
clean:
	@echo "[CLEAN] Removing build artifacts..."
	rm -f $(OBJS) $(TARGET)

run: $(TARGET)
	@echo "[RUN] Launching Scheduler..."
	./$(TARGET) ../experiment/task_set.csv 30 logs

debug: $(TARGET)
	@echo "[DEBUG] Running with cuda-gdb..."
	cuda-gdb ./$(TARGET) ../experiment/task_set.csv 30 logs

.PHONY: all clean run debug
