#include "GetCoreCount.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <locale>
#include <set>
#include <thread>
#include <vector>

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #define VC_EXTRALEAN
  #include <Windows.h>
  #include <malloc.h>
  #include <stdio.h>
  #include <tchar.h>
#endif  // _WIN32


#if defined(__APPLE__) || defined(__FreeBSD__)
  #include <sys/sysctl.h>
  #include <sys/types.h>
#endif
#if defined(__linux__)
  #include <unistd.h>

#endif

// trim from start (in place)
static inline void ltrim(std::string &s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) { return !std::isspace(ch); }));
}

// trim from end (in place)
static inline void rtrim(std::string &s) {
  s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(),
          s.end());
}

// // trim from both ends (in place)
// static inline void trim(std::string &s) {
//     ltrim(s);
//     rtrim(s);
// }


int ASSET::get_core_count() {

  int tcount = std::thread::hardware_concurrency();

#if defined(_WIN32)

  auto Run = [tcount]() {
    try {
      unsigned cores = 0;
      DWORD size = 0;
      GetLogicalProcessorInformation(NULL, &size);
      if (ERROR_INSUFFICIENT_BUFFER != GetLastError())
        return 0;
      const size_t Elements = size / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);

      std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer(Elements);
      if (GetLogicalProcessorInformation(&buffer.front(), &size) == FALSE)
        return 0;


      for (size_t i = 0; i < Elements; ++i) {
        if (buffer[i].Relationship == RelationProcessorCore)
          ++cores;
      }
      return (int(cores));

    } catch (...) {
      return tcount;
    }
  };

#elif defined(__linux__)

  auto Run = [tcount]() {
    /// I took this from boost threads and modified it to work without boost
    /// https://github.com/boostorg/thread/blob/develop/src/pthread/thread.cpp
    /// boost::thread::physical_concurrency


    auto trim = [](std::string &s) {
      s.erase(s.begin(),
              std::find_if(s.begin(), s.end(), [](unsigned char ch) { return !std::isspace(ch); }));

      s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(),
              s.end());
    };

    try {
      using namespace std;

      ifstream proc_cpuinfo("/proc/cpuinfo");
      const string physical_id("physical id"), core_id("core id");
      typedef std::pair<int, int> core_entry;  // [physical ID, core id]
      std::set<core_entry> cores;
      core_entry current_core_entry;

      string line;
      while (getline(proc_cpuinfo, line)) {
        if (line.empty())
          continue;

        size_t pos = line.find(":");
        if (pos == string::npos)
          return tcount;

        string key = line.substr(0, pos);
        string value = line.erase(0, pos + 1);
        trim(key);
        trim(value);

        if (key == physical_id) {
          current_core_entry.first = atoi(value.c_str());
          continue;
        }
        if (key == core_id) {
          current_core_entry.second = atoi(value.c_str());
          cores.insert(current_core_entry);
          continue;
        }
      }
      // Fall back to hardware_concurrency() in case
      // /proc/cpuinfo is formatted differently than we expect.
      return cores.size() != 0 ? int(cores.size()) : tcount;
    } catch (...) {
      return tcount;
    }
  };


#elif defined(__APPLE__)
  auto Run = [tcount]() {
    try {
      int count;
      size_t size = sizeof(count);
      sysctlbyname("hw.physicalcpu", &count, &size, NULL, 0) ? 0 : count;
      return count;
    } catch (...) {
      return tcount;
    }
  };

#else
  auto Run = [tcount]() { return tcount; };
#endif

  int ccount = std::max<int>(1, Run());
  ccount = std::min<int>(tcount, ccount);
  return ccount;
}
