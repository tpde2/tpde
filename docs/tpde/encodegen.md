\page tpde-encodegen EncodeGen Reference

# Overview

- writing instruction selection by hand is tedious, error-prone and not portable
- => autogenerate them from high-level code
- you write code, tool generates function that will encode the generated instructions trying to optimize them locally

# Class Hierarchy
- EncodeGen tool will generate a class that is added to your compiler as a mixin, i.e. becomes another base class
```
┌─────────┐         ┌────────────┐         ┌─────────┐ 
│IRAdaptor◄─ ─ ─ ─ ─┤CompilerBase├─ ─ ─ ─ ─►Assembler│ 
└─────────┘         └─────▲──────┘         └─────────┘ 
                          │                            
                          │                            
                  ┌───────┴────────┐                   
                  │UserCompilerBase│                   
                  └───────▲────────┘                   
                          │                            
                          │                            
                   ┌──────┴───────┐                    
                   │Compiler{Arch}│                    
                   └──────▲─▲─────┘                    
                          │ │                          
                          │ │  ┌──────────────────────┐
                          │ │  │ EncodeCompiler{Arch} │
                          │ │  └▲─┬───────────────────┘
                          │ │   │ │                    
                          │ │   │ │                    
                 ┌────────┴─┼───┴─┼─┐                  
                 │          └─────┘ │                  
                 │                  │                  
                 │UserCompiler{Arch}│                  
                 │                  │                  
                 │                  │                  
                 └──────────────────┘                  
```

# Snippets
- can generally be written in any language that compiles to LLVM-IR
- however, some constructs currently unsupported, e.g.
    - calling other functions
    - stack frames (i.e. large functions)
    - accessing globals
- will be transformed into function (snippet encoder) that will take one [GenericValuePart](@ref GenericValuePart) per input (register) and one [ScratchReg&](@ref ScratchReg)
    for each output (register)
- if input is constant ValuePartRef or an [GenericValuePart::Expr](@ref GenericValuePart::Expr), snippet encoder will attempt to fuse them into instructions if possible

```c
int addi32(int a, int b) {
    return a + b;
}
```

will be transformed into function conceptually like this (some implementation details omitted)
```cpp
void encode_addi32(GenericValuePart&& in0, GenericValuePart&& in1, ScratchReg& result) {
    // some setup
    ScratchReg internal{};

    AsmReg lhs_reg = internal.copy_try_reuse(in0);
    // try to fuse immediate operand
    if (in1.is_const() && is_imm32_encodable(in1.const_data())) {
        ASM(ADD32ri, lhs_reg, in1.const_data());
    } else {
        // it will actually also try to create an ADD32rm if the value is currently spilled
        AsmReg rhs_reg = in1.load_to_reg();
        ASM(ADD32rr, lhs_reg, rhs_reg);
    }

    result = std::move(internal);
}
```

# Compiling snippets
- First, have to compile to LLVM-IR
- For C on x86-64, you can use clang like so:

`clang -c -emit-llvm -ffreestanding -fcf-protection=none -O3 -fomit-frame-pointer -fno-math-errno --target=x86_64 -march=x86-64-v4 
 -o encode_template.bc <path/to/encode_template.c>`
- Then invoke the tpde_encodegen tool (either manually or from CMake by depending on tpde_encodegen) like so

`tpde_encodegen -o encode_template_x64.hpp encode_template.bc <additional ll or bc files>`
- This will create a file that contains an instance of `EncodeCompilerX64` that you can inherit in your compiler

# Embedding the EncodeCompiler
- simply add as base class to your existing compiler

> [!warning]
> you need to overwrite reset in your child class and call `Base::reset` as well as `EncodeCompiler{Arch}::reset`

so if you have
```cpp
#include <tpde/x64/CompilerX64.hpp>

struct MyCompilerX64 : tpde::CompilerX64<MyAdaptor, MyCompilerX64, CompilerBase, CompilerConfig>
```

you would simply add the EncodeCompiler:

```cpp
#include <tpde/x64/CompilerX64.hpp>
#include <encode_template_x64.hpp>

struct MyCompilerX64 : tpde::CompilerX64<MyAdaptor, MyCompilerX64, CompilerBase, CompilerConfig>,
    tpde_encodegen::EncodeCompiler<MyAdaptor, MyCompilerX64, CompilerBase, CompilerConfig>
```

# Calling Snippet Encoders
- simple as preparing the input ValueRefs and output ScratchReg, then calling them

```cpp
void compile_add(IRInstRef inst) {
    IRValueRef lhs_val = /* ... */;
    IRValueRef rhs_val = /* ... */;
    IRValueRef res_val = /* ... */;

    ValueRef lhs_ref = this->val_ref(lhs_val), rhs_ref = this->val_ref(rhs_val);
    ValueRef res_ref = this->result_ref(res_val);
    this->encode_addi32(lhs_ref.part(0), rhs_ref.part(0), res_ref.part(0));
}
```

## Passing Expressions
- on some architectures it is beneficial to merge address calculation into instructions
- for this purpose you can assemble `base_val + scale_imm * index_val + disp_imm` expressions using [GenericValuePart::Expr](@ref GenericValuePart::Expr)
  and pass them to the snippet encoders

so with the following snippet:
```c
void loadi8_zext(u8* ptr) {
    return (u32)*ptr;
}
```

you can write code like this:
```cpp
void compile_load(IRInstRef inst) {
    IRValueRef ptr_val = /* ... */;
    u32 load_off = /* ... */;
    IRValueRef res_val = /* ... */;

    auto [ptr_ref, ptr_part] = this->val_ref_single(ptr_val);

    GenericValuePart::Expr addr{};
    AsmReg base_reg = ptr_part.load_to_reg();

    // we want to reuse the register, e.g. for the load result, if it is salvageable
    if (ptr_part.can_salvage()) {
        ScratchReg scratch{this};
        scratch.alloc_specific(ptr_part.salvage());
        addr.base = std::move(scratch);
    } else {
        addr.base = base_reg;
    }

    // similar for scale and index if you wish to use them

    addr.disp = load_off;

    // call the snippet encoder with the calculated expression
    ValueRef res_ref = this->result_ref(res_val);
    this->encode_loadi8_zext(std::move(addr), res_ref.part(0));
}
```

which, if you have an IR load like `%1 = loadi8_zext %0, 20`

will generate assembly like this:
```
; assuming ptr is in rax
movzx eax, [rax + 20]
```

<div class="section_buttons">
 
| Previous          |                              Next |
|:------------------|----------------------------------:|
| [Compiler Reference](@ref tpde-compiler-ref) | [Guide: TPDE example back-end](@ref tpde-guide) |
 
</div>

