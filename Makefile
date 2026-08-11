# ============================================================================
# FlashGraph — CPU-Only Local Makefile
# ============================================================================
#
# Targets:
#   make all     — Build the smoke test binary (./flashgraph_main)
#   make test    — Build and run the arena unit tests (./test_arena)
#   make clean   — Remove all build artifacts
#
# This Makefile deliberately builds ONLY the CPU code. CUDA compilation
# is handled by CMakeLists.txt in the cloud environment (Phase 4).
# ============================================================================

CXX       := g++
CXXFLAGS  := -std=c++17 -O2 -Wall -Wextra -Wpedantic -march=native
INCLUDES  := -Iinclude

# Binaries
MAIN_BIN  := flashgraph_main
TEST_BIN  := test_arena

# ─── Default target ────────────────────────────────────────────────
.PHONY: all test clean

all: $(MAIN_BIN)

# ─── Object file rules (explicit to handle subdirectories) ─────────
src/arena_allocator.o: src/arena_allocator.cpp include/arena_allocator.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

src/cpu_ops.o: src/cpu_ops.cpp include/cpu_ops.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

src/main.o: src/main.cpp include/arena_allocator.h include/tensor.h include/cpu_ops.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

tests/test_arena.o: tests/test_arena.cpp include/arena_allocator.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# ─── Link main binary ──────────────────────────────────────────────
$(MAIN_BIN): src/main.o src/arena_allocator.o src/cpu_ops.o
	$(CXX) $(CXXFLAGS) $^ -o $@ -lm

# ─── Link test binary ──────────────────────────────────────────────
$(TEST_BIN): tests/test_arena.o src/arena_allocator.o
	$(CXX) $(CXXFLAGS) $^ -o $@ -lm

# ─── Run tests ─────────────────────────────────────────────────────
test: $(TEST_BIN)
	@echo ""
	@echo "Running arena tests..."
	@echo ""
	@./$(TEST_BIN)

# ─── Clean ──────────────────────────────────────────────────────────
clean:
	rm -f src/*.o tests/*.o
	rm -f $(MAIN_BIN) $(TEST_BIN)
