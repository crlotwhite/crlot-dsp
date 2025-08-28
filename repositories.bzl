load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def kissfft_dep():
    http_archive(
        name = "kissfft_src",
        urls = ["https://github.com/mborgerding/kissfft/archive/refs/tags/131.1.0.tar.gz"],
        strip_prefix = "kissfft-131.1.0",
        sha256 = "76c1aac87ddb7258f34b08a13f0eebf9e53afa299857568346aa5c82bcafaf1a",
        build_file_content = """
filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)
""",
    )
