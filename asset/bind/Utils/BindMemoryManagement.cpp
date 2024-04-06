#include <bind/Utils/BindMemoryManagement.h>

void ASSET::BindMemoryManager(py::module& m) {
  auto obj = py::class_<MemoryManager>(m, "MemoryManager");
  obj.def_static("enable_arena_memory", []() { MemoryManager::enable_arena_memory(); });
  obj.def_static("disable_arena_memory", []() { MemoryManager::disable_arena_memory(); });
}
