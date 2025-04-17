#pragma once
#include <cstddef>
#include <memory>
#include <type_traits>
#include <vector>

namespace arb::util {

template<typename T, size_t N = 8, typename NonReboundT = T>
struct small_buffer_vector_allocator{
    alignas(alignof(T)) std::byte buffer_[N * sizeof(T)];
    std::allocator<T> alloc_{};
    bool buffer_used_ = false;

    using value_type = T;
    using propagate_on_container_move_assignment = std::false_type;
    using propagate_on_container_swap = std::false_type;
    using is_always_equal = std::false_type;

    constexpr small_buffer_vector_allocator() noexcept = default;
    template<class U> constexpr small_buffer_vector_allocator(const small_buffer_vector_allocator<U, N, NonReboundT>&) noexcept {}
    template <class U> struct rebind { typedef small_buffer_vector_allocator<U, N, NonReboundT> other; };

    constexpr small_buffer_vector_allocator(const small_buffer_vector_allocator& other) noexcept : buffer_used_(other.buffer_used_) {}
    constexpr small_buffer_vector_allocator& operator=(const small_buffer_vector_allocator& other) noexcept { buffer_used_ = other.buffer_used_; return *this; }
    constexpr small_buffer_vector_allocator(small_buffer_vector_allocator&&) noexcept {}
    constexpr small_buffer_vector_allocator& operator=(const small_buffer_vector_allocator&&) noexcept { return *this; }

    [[nodiscard]] constexpr T* allocate(const size_t n) {
        // when the allocator was rebound we don't want to use the small buffer
        if constexpr (std::is_same_v<T, NonReboundT>) {
            if (n <= N) {
                buffer_used_ = true;
                // as long as we use less memory than the small buffer, we return a pointer to it
                return reinterpret_cast<T*>(&buffer_);
            }
        }
        buffer_used_ = false;
        //otherwise use the default allocator
        return alloc_.allocate(n);
    }
    constexpr void deallocate(void* p, const size_t n) {
      // we don't deallocate anything if the memory was allocated in small buffer
      if (&buffer_ != p) alloc_.deallocate(static_cast<T*>(p), n);
      buffer_used_ = false;
    }

    friend constexpr bool operator==(const small_buffer_vector_allocator& lhs, const small_buffer_vector_allocator& rhs) { return !lhs.buffer_used_ && !rhs.buffer_used_; }
    friend constexpr bool operator!=(const small_buffer_vector_allocator& lhs, const small_buffer_vector_allocator& rhs) { return !(lhs == rhs); }
};

template<typename T, size_t N = 8>
struct small_vector : public std::vector<T, small_buffer_vector_allocator<T, N>>{
    using vector_type = std::vector<T, small_buffer_vector_allocator<T, N>>;
    constexpr small_vector() noexcept { vector_type::reserve(N); }
    small_vector(const small_vector&) = default;
    small_vector& operator=(const small_vector&) = default;
    small_vector(small_vector&& other) noexcept(std::is_nothrow_move_constructible_v<T>) {
        if (other.size() <= N) vector_type::reserve(N);
        vector_type::operator=(std::move(other));
    }
    small_vector& operator=(small_vector&& other) noexcept(std::is_nothrow_move_constructible_v<T>) {
        if (other.size() <= N) vector_type::reserve(N);
        vector_type::operator=(std::move(other));
        return *this;
    }
    // use the default constructor first to reserve then construct the values
    explicit small_vector(size_t count): small_vector() { vector_type::resize(count); }
    small_vector(size_t count, const T& value): small_vector() { vector_type::assign(count, value); }
    template<typename It> small_vector(It first, It last): small_vector() { vector_type::insert(vector_type::begin(), first, last); }
    small_vector(std::initializer_list<T> init): small_vector() { vector_type::insert(vector_type::begin(), init); }
    friend void swap(small_vector& a, small_vector& b) noexcept {
        using std::swap;
        swap(static_cast<vector_type&>(a), static_cast<vector_type&>(b));
    }
};
} // arb::util
