"""Shim that forces the wheel to be tagged `py3-none-macosx_11_0_arm64`.

Project metadata lives in pyproject.toml; this file only customizes the
wheel tag. We ship a prebuilt Swift dylib + Metal kernel as package data,
so the wheel is platform-specific (Apple Silicon, macOS 11+) but otherwise
pure-Python — no compiled CPython extension. Default setuptools behavior
would either tag it `py3-none-any` (ignoring the platform constraint) or
`cpXY-cpXY-macosx_<build>_arm64` (locking to the Python build version),
both wrong. This shim picks the correct middle ground.
"""
from setuptools import setup
from wheel.bdist_wheel import bdist_wheel


class _PlatformWheel(bdist_wheel):
    def finalize_options(self) -> None:  # noqa: D401
        super().finalize_options()
        # Mark non-pure so a platform tag is emitted.
        self.root_is_pure = False

    def get_tag(self) -> tuple[str, str, str]:  # noqa: D401
        return ("py3", "none", "macosx_11_0_arm64")


setup(cmdclass={"bdist_wheel": _PlatformWheel})
