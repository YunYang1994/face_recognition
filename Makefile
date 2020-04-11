FLAGS= -Isrc/ `pkg-config --libs opencv` -Wall -Wno-unknown-pragmas -Wfatal-errors -fPIC -std=c++11

example01:
	g++ examples/example01.cpp src/image.cpp -o example01 ${example01}

.PHONY: clean
clean:
	rm -rf example01
