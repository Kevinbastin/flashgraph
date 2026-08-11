/**
 * @file test_arena.cpp
 * @brief Unit tests for the Arena allocator.
 *
 * Tests:
 *   1. Alignment correctness — every pointer is 64-byte aligned
 *   2. Sequential allocs don't overlap
 *   3. reset() reclaims space
 *   4. Over-allocation triggers abort (tested via fork)
 */

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <sys/wait.h>
#include <unistd.h>

#include "arena_allocator.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    static void test_##name(); \
    static struct Register_##name { \
        Register_##name() { test_##name(); } \
    } reg_##name; \
    static void test_##name()

#define ASSERT_TRUE(cond, msg) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "  FAIL: %s (line %d)\n", msg, __LINE__); \
            tests_failed++; \
            return; \
        } \
    } while(0)

#define PASS(name) \
    do { std::printf("  PASS: %s\n", name); tests_passed++; } while(0)

// ─── Test 1: Alignment ────────────────────────────────────────────

TEST(alignment) {
    flashgraph::Arena arena(4096);

    // Allocate several blocks of varying sizes
    void* p1 = arena.alloc(17);   // odd size
    void* p2 = arena.alloc(100);
    void* p3 = arena.alloc(1);    // tiny
    void* p4 = arena.alloc(256);

    ASSERT_TRUE((reinterpret_cast<uintptr_t>(p1) % 64) == 0,
                "p1 not 64-byte aligned");
    ASSERT_TRUE((reinterpret_cast<uintptr_t>(p2) % 64) == 0,
                "p2 not 64-byte aligned");
    ASSERT_TRUE((reinterpret_cast<uintptr_t>(p3) % 64) == 0,
                "p3 not 64-byte aligned");
    ASSERT_TRUE((reinterpret_cast<uintptr_t>(p4) % 64) == 0,
                "p4 not 64-byte aligned");

    PASS("alignment");
}

// ─── Test 2: No Overlap ───────────────────────────────────────────

TEST(no_overlap) {
    flashgraph::Arena arena(8192);

    void* p1 = arena.alloc(128);
    void* p2 = arena.alloc(256);
    void* p3 = arena.alloc(64);

    auto a1 = reinterpret_cast<uintptr_t>(p1);
    auto a2 = reinterpret_cast<uintptr_t>(p2);
    auto a3 = reinterpret_cast<uintptr_t>(p3);

    // p2 must start at or after p1 + 128 (with alignment padding)
    ASSERT_TRUE(a2 >= a1 + 128, "p2 overlaps p1");
    // p3 must start at or after p2 + 256
    ASSERT_TRUE(a3 >= a2 + 256, "p3 overlaps p2");

    PASS("no_overlap");
}

// ─── Test 3: Reset Reclaims Space ─────────────────────────────────

TEST(reset_reclaims) {
    flashgraph::Arena arena(1024);

    arena.alloc(512);
    ASSERT_TRUE(arena.used() >= 512, "used() should be >= 512 after alloc");

    arena.reset();
    ASSERT_TRUE(arena.used() == 0, "used() should be 0 after reset");
    ASSERT_TRUE(arena.remaining() == arena.capacity(),
                "remaining() should equal capacity after reset");

    // Should be able to allocate again
    void* p = arena.alloc(512);
    ASSERT_TRUE(p != nullptr, "alloc after reset returned null");
    ASSERT_TRUE((reinterpret_cast<uintptr_t>(p) % 64) == 0,
                "alloc after reset not aligned");

    PASS("reset_reclaims");
}

// ─── Test 4: Over-allocation aborts ───────────────────────────────

TEST(overalloc_aborts) {
    /**
     * We test that over-allocation calls abort() by forking a child
     * process and checking that it terminates with SIGABRT.
     *
     * This avoids killing the test process itself.
     */
    pid_t pid = fork();
    if (pid == 0) {
        // Child process: this should abort
        flashgraph::Arena arena(256);
        arena.alloc(512);  // exceeds capacity → abort()
        // If we reach here, the test failed
        _exit(0);
    }

    int status = 0;
    waitpid(pid, &status, 0);

    // SIGABRT = signal 6. WIFSIGNALED checks if terminated by signal.
    ASSERT_TRUE(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT,
                "over-allocation did not trigger SIGABRT");

    PASS("overalloc_aborts");
}

// ─── Main ──────────────────────────────────────────────────────────

int main() {
    std::printf("=== FlashGraph Arena Unit Tests ===\n\n");

    // Tests run automatically via static initialization above
    // (the TEST macro creates static struct instances)

    std::printf("\n--- Results: %d passed, %d failed ---\n",
                tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
