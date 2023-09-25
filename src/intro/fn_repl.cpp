#include <iostream>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cstdint>
#include <cstring>

void make_rwx (void* data, int size);
void force_print(void* fn_pointer, char* message);
void force_exit(void* fn_pointer, unsigned char exit_status);

void some_function() {
    std::cout << "Hello world!\n";
}

void another_function() {
    while(true){
        std::string str;
        std::cin >> str;
    }
}

int main() {

    force_print((void*)another_function,"Your message here\n");

    another_function();

    force_print((void*)another_function,"Yet another message\n");
    another_function();

    force_exit((void*)some_function,34);

    some_function();

    return 0;
}




// Makes a span of memory readable, writeable, and executable
void make_rwx (void* data, int size) {
    // First, we determine the page size, since we need a
    // page-aligned pointer
    int page_size = getpagesize();

    // Next, with some casting and arithmetic, we acheive our
    // desired alignment
    uintptr_t data_adr = (uintptr_t) data;
    uintptr_t align_adr = (data_adr/page_size)*page_size;
    void     *page = (void*) align_adr;

    // Expand our size to account for the offset
    size_t true_size = size + (data_adr - align_adr);

    // Change memory protections on the corresponding memory
    mprotect(page,true_size,PROT_READ | PROT_WRITE | PROT_EXEC);
}

// Alters the instructions of a function to make it immediately
// exit with the desired status.
// WARNING: This only works on x64 machines, and may write past
// the end of the original function if it has relatively few
// instructions to begin with
void force_exit(void* fn_pointer, unsigned char exit_status) {
    unsigned char exit_ops[18] = {
        0x48, 0xC7, 0xC0, 0x3C, 0x00, 0x00, 0x00, // rax, 0x3c
        0x48, 0xC7, 0xC7, 0x01, 0x00, 0x00, 0x00, // rdi, 0x??
        0x0F, 0x05 // syscall
    };

    exit_ops[10] = exit_status;

    make_rwx((void*)fn_pointer,18);
    memcpy((void*)fn_pointer,exit_ops,18);
}


// Alters the instructions of a function to make it print the
// provided message.
// WARNING: This only works on x64 machines, and may write past
// the end of the original function if it has relatively few
// instructions to begin with
void force_print(void* fn_pointer, char* message) {
    unsigned char print_ops[45] = {
        0x50,                                      // push rax
        0x57,                                      // push rdi
        0x56,                                      // push rsi
        0x52,                                      // push rdx
        0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00,  // mov  rax,0x1        -> perform write syscall
        0x48, 0xC7, 0xC7, 0x01, 0x00, 0x00, 0x00,  // mov  rdi,0x1        -> write to fd 1 (STDOUT)
        0x48, 0xBE, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00,                          // mov  rsi,0x???????? -> data pointer
        0x48, 0xBA, 0x78, 0x56, 0x34, 0x12, 0x78,
        0x56, 0x34, 0x12,                          // mov  rdx,0x???????? -> data size
        0x0F, 0x05,                                // syscall             -> perform call
        0x5a,                                      // pop  rdx
        0x5E,                                      // pop  rsi
        0x5F,                                      // pop  rdi
        0x58,                                      // pop  rax
        0xc3,                                      // ret      -> return to caller
    };

    uint64_t length = strlen(message);
    memcpy(print_ops+20,&message,8);    // inject message pointer
    memcpy(print_ops+30,&length ,8);    // inject message length

    make_rwx((void*)fn_pointer,45);
    memcpy((void*)fn_pointer,print_ops,45);
}
