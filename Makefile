TARGET := mnistCNN
SRC    := main.cpp
LIBS_NAME := armadillo mlpack boost_serialization

CXX := g++
CXXFLAGS += -std=c++11 -Wall -Wextra -O3 -DNDEBUG -fopenmp
LDFLAGS += -fopenmp
LDFLAGS += -L .
INCLFLAGS  := -I .
CXXFLAGS += $(INCLFLAGS)

OBJS := $(SRC:.cpp=.o)
LIBS := $(addprefix -l, $(LIBS_NAME))
CLEAN_LIST := $(TARGET) $(OBJS)

default: all

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET) $(LDFLAGS) $(LIBS)

.PHONY: all
all: $(TARGET)

.PHONY: clean
clean:
	@echo CLEAN $(CLEAN_LIST)
	@rm -f $(CLEAN_LIST)