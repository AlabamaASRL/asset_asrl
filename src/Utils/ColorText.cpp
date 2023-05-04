#include "ColorText.h"
#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #define VC_EXTRALEAN
  #include <Windows.h>
  #include <stdio.h>
#endif  // _WIN32

#ifndef ENABLE_PROCESSED_OUTPUT
  // From MSDN about BOOL SetConsoleMode(HANDLE, DWORD).
  #define ENABLE_PROCESSED_OUTPUT 0x0001
#endif  // not ENABLE_PROCESSED_OUTPUT

#ifndef ENABLE_WRAP_AT_EOL_OUTPUT
  // From MSDN about BOOL SetConsoleMode(HANDLE, DWORD).
  #define ENABLE_WRAP_AT_EOL_OUTPUT 0x0002
#endif  // not ENABLE_WRAP_AT_EOL_OUTPUT

#ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
  // From MSDN about BOOL SetConsoleMode(HANDLE, DWORD).
  #define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
#endif  // not ENABLE_VIRTUAL_TERMINAL_PROCESSING


void ASSET::enable_color_console() {

  // Only neccessary on windows
#if defined(_WIN32)
  HANDLE handle = GetStdHandle(STD_OUTPUT_HANDLE);
  DWORD mode = 0;
  if (handle != INVALID_HANDLE_VALUE) {
    if (!GetConsoleMode(handle, &mode)) {
      printf("  GetConsoleMode 1 fail, err: %d\n", GetLastError());
      return;
    }

    if (!SetConsoleMode(handle,
                        mode | ENABLE_PROCESSED_OUTPUT | ENABLE_WRAP_AT_EOL_OUTPUT
                            | ENABLE_VIRTUAL_TERMINAL_PROCESSING)) {
      printf("  SetConsoleMode fail, err: %d\n", GetLastError());
      return;
    }
    if (!GetConsoleMode(handle, &mode)) {
      printf("  GetConsoleMode 2 fail, err: %d\n", GetLastError());
      return;
    }
  }
#endif
}