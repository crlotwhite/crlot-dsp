.PHONY: all clean run

all: clean build run

build:
	bazel build //main:main

clean:
	rm test_output.wav
	bazel clean

run:
	bazel-bin/main/main

test:
	bazel test --test_output=all //tests:hello_test

# 	curl -L https://github.com/mborgerding/kissfft/archive/refs/tags/131.1.0.tar.gz | shasum -a 256 | cut -d' ' -f1