
CC := @g++
ECHO := @echo
SRCDIR := src
OBJDIR := objs
BINDIR := ./
OUTNAME = onnx_to_trt.so
LEAN := /data/yyh/classify/lean

CFLAGS := -std=c++11 -shared -fPIC -g -O3 -fopenmp -w
CUFLAGS := -std=c++11 -g -O3 -w
INC_OPENCV := $(LEAN)/opencv4.2.0/include/opencv4 $(LEAN)/opencv4.2.0/include/opencv4/opencv $(LEAN)/opencv4.2.0/include/opencv4/opencv2
INC_LOCAL := ./src
INC_SYS := 
INC_CUDA := /usr/local/cuda-10.0/include 
INC_BOOST := $(LEAN)/boost/include
INC_PYTHON := /data/yyh/anaconda3/include/python3.7m
INC_TENSORRT := $(LEAN)/tensorRTIntegrate
INCS := $(INC_OPENCV) $(INC_LOCAL) $(INC_SYS) $(INC_CUDA) $(INC_BOOST) $(INC_PYTHON) $(INC_TENSORRT)
INCS := $(foreach inc, $(INCS), -I$(inc))

LIB_CUDA := $(LEAN)/cudnn7.6.5.32-cuda10.2
LIB_SYS := 
LIB_BOOST := $(LEAN)/boost/lib
LIB_OPENCV := $(LEAN)/opencv4.2.0/lib
LIB_PYTHON := /data/yyh/anaconda3/lib
LIB_TENSORRT := $(LEAN)/tensorRTIntegrate
LIBS := $(LIB_SYS) $(LIB_CUDA) $(LIB_OPENCV) $(LIB_BOOST) $(LIB_PYTHON) $(LIB_TENSORRT)
RPATH := $(foreach lib, $(LIBS),-Wl,-rpath $(lib))
LIBS := $(foreach lib, $(LIBS),-L$(lib))

LD_OPENCV := opencv_core opencv_highgui opencv_imgproc opencv_video opencv_videoio opencv_imgcodecs
LD_BOOST := boost_filesystem boost_python37 boost_thread
LD_NVINFER := 
LD_CUDA := cudnn
LD_SYS := dl pthread stdc++
LD_PYTHON := python3.7m
LD_TENSORRT := tensorRTIntegrate
LDS := $(LD_OPENCV) $(LD_NVINFER) $(LD_CUDA) $(LD_SYS) $(LD_PYTHON) $(LD_BOOST) $(LD_TENSORRT)
LDS := $(foreach lib, $(LDS), -l$(lib))

SRCS := $(shell cd $(SRCDIR) && find -name "*.cpp")
OBJS := $(patsubst %.cpp,%.o,$(SRCS))
OBJS := $(foreach item,$(OBJS),$(OBJDIR)/$(item))
CUS := $(shell cd $(SRCDIR) && find -name "*.cu")
CUOBJS := $(patsubst %.cu,%.o,$(CUS))
CUOBJS := $(foreach item,$(CUOBJS),$(OBJDIR)/$(item))
OBJS := $(subst /./,/,$(OBJS))
CUOBJS := $(subst /./,/,$(CUOBJS))

all : $(BINDIR)/$(OUTNAME)
	$(ECHO) Compile done.

run: all
	@cd $(BINDIR) && ./$(OUTNAME)

$(BINDIR)/$(OUTNAME): $(OBJS) $(CUOBJS)
	$(ECHO) Linking: $@
	@$(CC) $(CFLAGS) $(LIBS) -o $@ $^ $(LDS) $(RPATH)

$(CUOBJS) : $(OBJDIR)/%.o : $(SRCDIR)/%.cu
	@if [ ! -d $@ ]; then mkdir -p $(dir $@); fi
	$(ECHO) Compiling: $<
	@$(CUCC) $(CUFLAGS) $(INCS) -c -o $@ $<

$(OBJS) : $(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	@if [ ! -d $@ ]; then mkdir -p $(dir $@); fi
	$(ECHO) Compiling: $<
	@$(CC) $(CFLAGS) $(INCS) -c -o $@ $<

clean:
	rm -rf $(OBJDIR) $(BINDIR)/$(OUTNAME)
