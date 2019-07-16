CXX:=g++
CXXFLAGS:=-std=c++11 -MD -Wall -g
#LDFLAGS:=$(shell pkg-config --libs opencv)

BIN:=ocr.exe
SOURCE:=$(wildcard *.cc)
OBJECT:=$(SOURCE:%.cc=%.o)
DEPEND:=$(SOURCE:%.cc=%.d)

all: $(BIN)
ocr.exe: $(SOURCE)
	$(CXX) $^ -o $(BIN) $(LDFLAGS)


clean:
	-rm $(OBJECT) $(DEPEND) $(BIN)
.PHONY: clean
