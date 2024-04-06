#include <ASSET/Utils/MemoryManagement.h>

namespace ASSET {

  thread_local MemoryManager::SuperScalarStackType MemoryManager::SuperScalarStack =
      detail::TempStack<DefaultSuperScalar>(ASSET_DEFAULT_ARENA_SIZE);
  thread_local MemoryManager::ScalarStackType MemoryManager::ScalarStack =
      detail::TempStack<double>(ASSET_DEFAULT_ARENA_SIZE);
  bool MemoryManager::UseArena = true;

}  // namespace ASSET
