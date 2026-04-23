# SPDX-License-Identifier: Apache-2.0
# Homebrew formula for vllm-swift
#
# Install: brew install TheTom/tap/vllm-swift
# Or:      brew tap TheTom/tap && brew install vllm-swift
#
# This formula builds the Swift/Metal bridge dylib and installs a
# wrapper script. The Python vLLM plugin is installed separately
# via pip (Homebrew doesn't manage Python venvs well).

class VllmSwift < Formula
  desc "Native Swift/Metal backend for vLLM on Apple Silicon"
  homepage "https://github.com/TheTom/vllm-swift"
  url "https://github.com/TheTom/vllm-swift.git", branch: "main"
  version "0.1.0"
  license "Apache-2.0"

  depends_on xcode: ["15.0", :build]
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

    # Install the Python plugin source
    (libexec/"vllm_swift").install Dir["vllm_swift/*.py"]
    libexec.install "pyproject.toml"

    # Install scripts
    (libexec/"scripts").install Dir["scripts/*"]

    # Create wrapper script that sets DYLD_LIBRARY_PATH automatically
    (bin/"vllm-swift").write <<~EOS
      #!/usr/bin/env bash
      # vllm-swift wrapper — sets up dylib path and runs vllm serve
      #
      # Usage: vllm-swift serve <model> [args...]
      #        vllm-swift setup   — install Python plugin into current venv
      #        vllm-swift test    — run integration test

      export DYLD_LIBRARY_PATH="#{lib}:${DYLD_LIBRARY_PATH:-}"

      case "${1:-}" in
        setup)
          echo "Installing vllm-swift Python plugin..."
          if [ -z "${VIRTUAL_ENV:-}" ]; then
            echo "WARNING: No active virtualenv. Consider: python3 -m venv .venv && source .venv/bin/activate"
          fi
          pip install -e "#{libexec}" 2>&1
          echo ""
          echo "Done. Run: vllm-swift serve <model_path>"
          ;;
        test)
          shift
          exec "#{libexec}/scripts/integration_test.sh" "$@"
          ;;
        serve)
          shift
          exec python3 -m vllm.entrypoints.openai.api_server "$@"
          ;;
        completions|chat)
          # Pass through to vllm CLI
          exec python3 -m vllm.entrypoints.openai.api_server "$@"
          ;;
        *)
          if [ $# -eq 0 ]; then
            echo "vllm-swift — Native Swift/Metal backend for vLLM"
            echo ""
            echo "Usage:"
            echo "  vllm-swift setup              Install Python plugin into current venv"
            echo "  vllm-swift serve <model> ...   Start OpenAI-compatible API server"
            echo "  vllm-swift test [model]        Run integration test"
            echo ""
            echo "First time setup:"
            echo "  python3 -m venv .venv && source .venv/bin/activate"
            echo "  pip install vllm"
            echo "  vllm-swift setup"
            echo "  vllm-swift serve ~/models/Qwen3-0.6B-4bit --max-model-len 2048"
            echo ""
            echo "dylib: #{lib}/libVLLMBridge.dylib"
          else
            # Unknown command — pass everything to vllm
            exec python3 -m vllm.entrypoints.openai.api_server "$@"
          fi
          ;;
      esac
    EOS
  end

  def caveats
    <<~EOS
      vllm-swift requires a Python virtualenv with vLLM installed.

      First time setup:
        python3 -m venv .venv
        source .venv/bin/activate
        pip install vllm
        vllm-swift setup

      Then serve a model:
        vllm-swift serve ~/models/Qwen3-0.6B-4bit --max-model-len 2048

      Models can be downloaded from HuggingFace:
        huggingface-cli download mlx-community/Qwen3-0.6B-4bit --local-dir ~/models/Qwen3-0.6B-4bit
    EOS
  end

  test do
    # Verify dylib loads
    assert_predicate lib/"libVLLMBridge.dylib", :exist?
    # Verify wrapper works
    assert_match "vllm-swift", shell_output("#{bin}/vllm-swift")
  end
end
