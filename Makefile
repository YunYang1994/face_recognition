VPATH=./src/:./examples # 如果某个文件找不到，就会去指定的目录寻找它
FLAGS= -Iinclude -Isrc/ `pkg-config --libs opencv` -Wall -Wno-unknown-pragmas -Wfatal-errors -fPIC -std=c++11

example01:
	g++ examples/example01.cpp src/image.cpp -o example01 ${FLAGS}
example02:
	g++ examples/example02.cpp src/image.cpp -o example02 ${FLAGS}

.PHONY: clean
clean:
	rm -rf example01 example02
