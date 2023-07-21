#pragma once

#include <spdlog/common.h>
#include <spdlog/sinks/base_sink.h>

#include <algorithm>

namespace ASSET {

  // Color Removing File Sink ////////////////////////////////////////////////////////////////////////////////

  template<typename Mutex>
  class style_removing_file_sink : public spdlog::sinks::base_sink<Mutex> {
   public:
    explicit style_removing_file_sink(const spdlog::filename_t &filename,
                                      bool truncate = false,
                                      const spdlog::file_event_handlers &event_handlers = {})
        : file_helper_ {event_handlers} {
      file_helper_.open(filename, truncate);
    }
    const spdlog::filename_t &filename() const {
      return file_helper_.filename();
    }

   protected:
    void sink_it_(const spdlog::details::log_msg &msg) override {
      spdlog::memory_buf_t formatted;
      spdlog::sinks::base_sink<Mutex>::formatter_->format(msg, formatted);

      // Remove style
      int nAfter, nRemove;
      char *mPtr;
      char *styleCharPtr = std::find(formatted.begin(), formatted.end(), '\033');
      while (styleCharPtr != formatted.end()) {
        mPtr = std::find(styleCharPtr, formatted.end(), 'm');
        nRemove = mPtr - styleCharPtr + 1;
        nAfter = formatted.end() - mPtr;
        for (int i = 0; i < nAfter; i++) {
          *(styleCharPtr + i) = *(styleCharPtr + i + nRemove);
        }
        formatted.resize(formatted.size() - nRemove);

        styleCharPtr = std::find(formatted.begin(), formatted.end(), '\033');
      }

      file_helper_.write(formatted);
    }
    void flush_() override {
      file_helper_.flush();
    }

   private:
    spdlog::details::file_helper file_helper_;
  };

  // Useful Types ////////////////////////////////////////////////////////////////////////////////////////////

  using style_removing_file_sink_mt = style_removing_file_sink<std::mutex>;
  using style_removing_file_sink_st = style_removing_file_sink<spdlog::details::null_mutex>;

}  // namespace ASSET
