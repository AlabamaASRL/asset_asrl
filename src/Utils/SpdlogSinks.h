#pragma once

#include <spdlog/common.h>
#include <spdlog/sinks/base_sink.h>

#include <algorithm>

namespace ASSET {

  // Color Removing File Sink ////////////////////////////////////////////////////////////////////////////////

  template<typename Mutex>
  class color_removing_file_sink : public spdlog::sinks::base_sink<Mutex> {
   public:
    explicit color_removing_file_sink(const spdlog::filename_t &filename,
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

      // Remove color
      int nAfter, nRemove;
      char *mPtr;
      char *colorCharPtr = std::find(formatted.begin(), formatted.end(), '\033');
      while (colorCharPtr != formatted.end()) {
        mPtr = std::find(colorCharPtr, formatted.end(), 'm');
        nRemove = mPtr - colorCharPtr + 1;
        nAfter = formatted.end() - mPtr;
        for (int i = 0; i < nAfter; i++) {
          *(colorCharPtr + i) = *(colorCharPtr + i + nRemove);
        }
        formatted.resize(formatted.size() - nRemove);
        // formatted.size_ = formatted.size_ - nRemove;

        colorCharPtr = std::find(formatted.begin(), formatted.end(), '\033');
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

  using color_removing_file_sink_mt = color_removing_file_sink<std::mutex>;
  using color_removing_file_sink_st = color_removing_file_sink<spdlog::details::null_mutex>;

}  // namespace ASSET
