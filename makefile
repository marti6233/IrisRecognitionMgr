# makefile for image processing
all : image.cpp processing.cpp eye.cpp
	g++ -o imagetest -fopenmp image.cpp processing.cpp eye.cpp `pkg-config opencv4 --cflags --libs`

clean : imagetest
	rm *[-o]
