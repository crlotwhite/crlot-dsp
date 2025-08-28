load("//:repositories.bzl", "kissfft_dep")

def _non_module_deps_impl(module_ctx):
    kissfft_dep()

non_module_deps = module_extension(
    implementation = _non_module_deps_impl,
)
