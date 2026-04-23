# Releasing vllm-swift

## Release Process

### 1. Update mlx-swift-lm stable branch

When alpha has changes needed by vllm-swift:

```bash
cd ~/dev/mlx-swift-lm
git push fork alpha:vllm-swift-stable --force
```

This updates the pinned snapshot that SPM pulls. No version bump needed — it's a branch ref.

### 2. Build and test locally

```bash
cd ~/dev/vllm-swift

# Build Swift bridge
cd swift && swift build -c release && cd ..

# Run tests
python3 -m pytest tests/ --cov=vllm_swift --cov-fail-under=95 -v

# Lint
ruff check vllm_swift/ tests/ && ruff format --check vllm_swift/ tests/
```

### 3. Update version (if needed)

Update version in:
- `pyproject.toml` → `version = "X.Y.Z"`
- `homebrew/vllm-swift.rb` → `version "X.Y.Z"`
- `scripts/build_bottle.sh` → `VERSION="X.Y.Z"`
- Wrapper script version string in `build_bottle.sh`

### 4. Commit and push

```bash
git add -A
git commit -m "release: vX.Y.Z"
git push origin main
git tag vX.Y.Z
git push origin vX.Y.Z
```

### 5. Build and upload Homebrew bottle

```bash
./scripts/build_bottle.sh
```

This script:
1. Builds the Swift bridge (release)
2. Packages dylib + Python plugin + wrapper into a bottle tarball
3. Uploads to `TheTom/homebrew-tap` GitHub Releases
4. Prints the SHA256 for the formula

### 6. Update tap formula

Copy the bottle SHA from step 5 output into the tap formula:

```bash
cd /tmp/homebrew-tap  # or wherever you have the tap cloned
# Update Formula/vllm-swift.rb with new SHA
git add -A && git commit -m "bottle: update to vX.Y.Z" && git push origin main
```

Or if you don't have it cloned:

```bash
git clone https://github.com/TheTom/homebrew-tap.git /tmp/homebrew-tap
# Edit Formula/vllm-swift.rb — update sha256 line
cd /tmp/homebrew-tap && git add -A && git commit -m "bottle: update to vX.Y.Z" && git push
```

### 7. Test on a clean machine

```bash
ssh toms-mac-mini.local
export PATH=/opt/homebrew/bin:$PATH
brew uninstall vllm-swift
brew untap TheTom/tap
rm -rf $(brew --cache)/downloads/*vllm*
brew tap TheTom/tap && brew install vllm-swift
vllm-swift version
```

## Quick Reference

| What | Where |
|------|-------|
| Swift bridge source | `swift/Sources/VLLMBridge/Bridge.swift` |
| Python plugin | `vllm_swift/` |
| mlx-swift-lm dep | `TheTom/mlx-swift-lm` branch `vllm-swift-stable` |
| Homebrew tap | `TheTom/homebrew-tap` |
| Bottle tarball | GitHub Releases on `TheTom/homebrew-tap` tag `bottles` |
| Build bottle locally | `./scripts/build_bottle.sh` |
| Mac Mini test machine | `ssh toms-mac-mini.local` |

## Updating mlx-swift-lm dependency

The `swift/Package.swift` points to:
```swift
.package(url: "https://github.com/TheTom/mlx-swift-lm.git", branch: "vllm-swift-stable")
```

To pick up new changes:
1. Push alpha → vllm-swift-stable (step 1 above)
2. In vllm-swift: `cd swift && swift package update`
3. Build, test, commit the updated `Package.resolved`
4. Rebuild bottle (step 5)

## Git conventions

- Commits: `Co-Authored-By: tturney@psyguard.ai`
- Email: `tturney1@gmail.com` (TheTom repos)
- Don't push to main without tests passing
