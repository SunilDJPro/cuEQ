CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -Iinclude
NVCCFLAGS = -std=c++17 -O3 -Iinclude
LDFLAGS = -lcufft -lcudart

SRCDIR = src
INCDIR = include
OBJDIR = obj
BINDIR = bin

SOURCES_CXX = $(wildcard $(SRCDIR)/*.cpp)
SOURCES_CU = $(wildcard $(SRCDIR)/*.cu)
OBJECTS_CXX = $(SOURCES_CXX:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)
OBJECTS_CU = $(SOURCES_CU:$(SRCDIR)/%.cu=$(OBJDIR)/%.o)
OBJECTS = $(OBJECTS_CXX) $(OBJECTS_CU)

TARGET = $(BINDIR)/cuEQ

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS) | $(BINDIR)
	$(NVCC) $(OBJECTS) -o $@ $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cu | $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(BINDIR):
	mkdir -p $(BINDIR)

clean:
	rm -rf $(OBJDIR) $(BINDIR)