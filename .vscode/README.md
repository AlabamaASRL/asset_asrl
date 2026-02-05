# VSCode Configuration for ASSET

This directory contains VSCode configuration files to help with building ASSET using CMake Tools.

## Files

### `cmake-kits.json`
Defines CMake kits for building with LLVM/Clang on Windows and Unix-like systems.

**Available Kits:**
- **Clang-cl (LLVM)**: For Windows x64 builds using LLVM clang-cl
- **Clang-cl ARM64 (LLVM)**: For Windows ARM64 builds using LLVM clang-cl
- **Clang (Unix-like)**: For Linux/macOS builds using system clang

### `settings.json`
Workspace settings for CMake Tools, C++ IntelliSense, and general editor configuration.

### `extensions.json`
Recommended VSCode extensions for working with this project.

## Usage

1. **Install LLVM** (Windows only):
   - Download from: https://github.com/llvm/llvm-project/releases
   - Install to the default location: `C:\Program Files\LLVM\`
   - Ensure `clang-cl.exe`, `llvm-ar.exe`, `llvm-ranlib.exe`, and `lld-link.exe` are in the bin directory

2. **Install Ninja** (all platforms):
   - Windows: `choco install ninja` or download from https://ninja-build.org/
   - Linux: `sudo apt install ninja-build`
   - macOS: `brew install ninja`

3. **Select CMake Kit**:
   - Open the Command Palette (Ctrl+Shift+P / Cmd+Shift+P)
   - Run: "CMake: Select a Kit"
   - Choose the appropriate kit:
     - Windows x64: "Clang-cl (LLVM)"
     - Windows ARM64: "Clang-cl ARM64 (LLVM)"
     - Linux/macOS: "Clang (Unix-like)"

4. **Configure and Build**:
   - Run: "CMake: Configure"
   - Run: "CMake: Build"

## Customization

If your LLVM installation is in a different location, edit the paths in `cmake-kits.json`:
- Update `compilers.C` and `compilers.CXX`
- Update all `cmakeSettings` paths to match your installation

For ARM64 builds, you may also need to:
- Install ARM64 Python (see README.md)
- Install OpenBLAS via vcpkg: `vcpkg install openblas:arm64-windows`
- Set `Python_ROOT_DIR` in CMake settings if needed

## Troubleshooting

### "Kit not found" error
- Verify LLVM is installed at `C:\Program Files\LLVM\`
- Check that `clang-cl.exe` exists in the bin directory
- Reload the VSCode window after installing LLVM

### "Ninja not found" error
- Install Ninja build system
- Ensure it's in your PATH
- Restart VSCode after installation

### IntelliSense not working
- Ensure the C/C++ extension is installed
- Run "CMake: Configure" at least once
- Check that `"C_Cpp.default.configurationProvider": "ms-vscode.cmake-tools"` is set in settings

## More Information

See the main project README.md for detailed build instructions and platform-specific requirements.
