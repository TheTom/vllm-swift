# SPDX-License-Identifier: Apache-2.0
# Homebrew formula for vllm-swift
#
# Install: brew install TheTom/tap/vllm-swift
# Or:      brew tap TheTom/tap && brew install vllm-swift
#
# After install, just run: vllm-swift serve <model>
# Everything (dylib, venv, vLLM, plugin) is handled automatically.

class VllmSwift < Formula
  desc "Native Swift/Metal backend for vLLM on Apple Silicon"
  homepage "https://github.com/TheTom/vllm-swift"
  url "https://github.com/TheTom/vllm-swift.git", branch: "main"
  version "0.1.0"
  license "Apache-2.0"

  depends_on xcode: ["15.0", :build]
  depends_on "python@3.12"
  depends_on :macos
  depends_on arch: :arm64

  def install
    # Build Swift bridge (release)
    cd "swift" do
      system "swift", "build", "-c", "release"
    end

    build_dir = "swift/.build/arm64-apple-macosx/release"

    # Install the dylib
    lib.install "#{build_dir}/libVLLMBridge.dylib"

    # Copy metallib if it exists
    metallib = "#{build_dir}/mlx.metallib"
    lib.install metallib if File.exist?(metallib)

    # Install the Python plugin source + pyproject.toml
    libexec.install "pyproject.toml"
    (libexec/"vllm_swift").install Dir["vllm_swift/*.py"]

    # Install scripts
    (libexec/"scripts").install Dir["scripts/*"]

    # Create managed venv with vLLM + plugin pre-installed
    venv_dir = libexec/"venv"
    system "python3", "-m", "venv", venv_dir
    venv_pip = venv_dir/"bin/pip"
    venv_python = venv_dir/"bin/python3"

    # Install torch (CPU wheel — Metal acceleration comes from Swift side)
    system venv_pip, "install", "-q",
           "torch", "--index-url", "https://download.pytorch.org/whl/cpu"

    # Install vLLM
    system venv_pip, "install", "-q", "vllm>=0.19.0"

    # Install the plugin
    system venv_pip, "install", "-q", "-e", libexec

    # Create wrapper that uses the managed venv
    (bin/"vllm-swift").write <<~EOS
      #!/usr/bin/env bash
      # vllm-swift — Native Swift/Metal LLM inference
      #
      # Usage:
      #   vllm-swift serve <model> [vllm args...]
      #   vllm-swift download <hf-model-id>
      #   vllm-swift test [model_path]
      #   vllm-swift version

      export DYLD_LIBRARY_PATH="#{lib}:${DYLD_LIBRARY_PATH:-}"
      VENV_PYTHON="#{venv_dir}/bin/python3"

      case "${1:-}" in
        serve)
          shift
          exec "$VENV_PYTHON" -m vllm.entrypoints.openai.api_server "$@"
          ;;
        download)
          shift
          MODEL="${1:?Usage: vllm-swift download <model-id>}"
          SHORT="$(basename "$MODEL")"
          echo "Downloading $MODEL to ~/models/$SHORT..."
          exec "$VENV_PYTHON" -c "
      from huggingface_hub import snapshot_download
      import os
      path = snapshot_download('$MODEL', local_dir=os.path.expanduser('~/models/$SHORT'))
      print(f'Downloaded to {path}')
      "
          ;;
        test)
          shift
          exec "#{libexec}/scripts/integration_test.sh" "$@"
          ;;
        version)
          echo "vllm-swift 0.1.0"
          echo "dylib: #{lib}/libVLLMBridge.dylib"
          "$VENV_PYTHON" -c "import vllm; print(f'vLLM: {vllm.__version__}')" 2>/dev/null || true
          ;;
        *)
          echo "vllm-swift — Native Swift/Metal backend for vLLM on Apple Silicon"
          echo ""
          echo "Usage:"
          echo "  vllm-swift serve <model> [args]   Start OpenAI-compatible API server"
          echo "  vllm-swift download <model-id>    Download model from HuggingFace"
          echo "  vllm-swift test [model_path]      Run integration test"
          echo "  vllm-swift version                Show version info"
          echo ""
          echo "Examples:"
          echo "  vllm-swift download mlx-community/Qwen3-0.6B-4bit"
          echo "  vllm-swift serve ~/models/Qwen3-0.6B-4bit --max-model-len 2048"
          echo "  vllm-swift serve ~/models/Qwen3-4B-4bit --max-model-len 4096 --port 8080"
          ;;
      esac
    EOS
  end

  def caveats
    <<~EOS
      vllm-swift is ready to use. No additional setup needed.

      Download a model and serve:
        vllm-swift download mlx-community/Qwen3-0.6B-4bit
        vllm-swift serve ~/models/Qwen3-0.6B-4bit --max-model-len 2048

      The server exposes an OpenAI-compatible API at http://localhost:8000
    EOS
  end

  test do
    assert_predicate lib/"libVLLMBridge.dylib", :exist?
    assert_match "vllm-swift", shell_output("#{bin}/vllm-swift")
    assert_match "0.1.0", shell_output("#{bin}/vllm-swift version")
  end
end
