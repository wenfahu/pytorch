#pragma once

#include <torch/csrc/distributed/autograd/context/context.h>

namespace torch {
namespace distributed {
namespace autograd {

// This class stores a weak_ptr to the current dist_autograd context in a thread
// local variable so that a reader can retrieve the context.
class ThreadLocalDistAutogradContext {
 public:
  explicit ThreadLocalDistAutogradContext(ContextWeakPtr&& context_wp);
  ~ThreadLocalDistAutogradContext();

  static ContextWeakPtr getContextWeakPtr();

 private:
  ContextWeakPtr prev_context_weak_ptr_;
  ContextWeakPtr context_weak_ptr_;
};

} // namespace autograd
} // namespace distributed
} // namespace torch
