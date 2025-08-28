.PHONY: all clean run

all: clean build run

build:
	bazel $(RC) build //main:main

clean:
	bazel clean

run:
	bazel-bin/main/main$(EXEEXT)