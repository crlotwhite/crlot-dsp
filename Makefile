.PHONY: all clean run

all: clean build run

build:
	bazel build //main:main

clean:
	bazel clean

run:
	bazel-bin/main/main

test:
	bazel test --test_output=all //tests:hello_test