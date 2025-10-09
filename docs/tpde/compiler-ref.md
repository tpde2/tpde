\page tpde-compiler-ref Compiler Reference

This is the reference for the `CompilerBase` and related classes. It is the main driver of compilation and the class you as the user will
in the end create an instance of to start compilation.

The reference starts by explaining the basic class hierarchy used in the compiler, ...

[TOC]

## Class Hierarchy
- Compiler uses static polymorphism, i.e. templates using CRTP to adapt the compiler to your IR and use your provided functions
- CompilerBase takes three template parameters
  - Adaptor class type
  - final derived type
  - config for Assembler type, some flags and architecture-specific information
- architecture-specific Compiler class (currently x64/arm64) which takes 4 template parameters
  - Adaptor class type
  - final derived type
  - base compiler type (can be CompilerBase or a user-provided base class that inherits from CompilerBase that can be used for architecture-independent functionality)
  - config for CompilerBase config and some architecture-specific config
- final compiler class which inherits from the architecture-specific compiler class and provides the template types
- many functions are called in the base classes using the derived class giving you the option to override or inject custom behavior

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
                   └──────▲───────┘                   
                          │                           
                          │                           
                 ┌────────┴─────────┐                 
                 │UserCompiler{Arch}│                 
                 └──────────────────┘               
```

## Config
- type which provides typedefs and constants used to configure the compiler
- mostly internal or currently not useful (defined in CompilerConfig.hpp/CompilerX64.hpp/CompilerA64.hpp)
- e.g. provides typedef `Assembler` which tells the compiler which assembler to use, but currently only one assembler supported for each architecture
- most interesting option is `DEFAULT_VAR_REF_HANDLING`, which, when turned off, allows to setup custom variable refs, e.g. for globals (see later).

## Required functions to implement
- concept described in Compiler.hpp / CompilerX64.hpp / CompilerA64.hpp

### General

#### cur_func_may_emit_calls
```cpp
bool cur_func_may_emit_calls() const noexcept;
```
Can the compilation of any instruction in the current function emit calls? This may be used to optimize register allocation or stack frame setup. If you can cheaply provide the answer,
you should. Otherwise you currently should always return `true`.

#### cur_personality_func
```cpp
SymRef cur_personality_func() noexcept;
```
Returns the personality function of the current function. This is relevant for exception handling
and cleanup actions performed on unwinding.

#### try_force_fixed_assignment
```cpp
bool try_force_fixed_assignment(IRValueRef val) const noexcept;
```
Should the register allocator try to allocate a fixed register for `val` even if its heuristic tells it not to.
Mostly useful for debugging, you should return `false`.

#### val_parts
```cpp
struct ValParts {
    u32 count() const noexcept;
    u32 size_bytes(u32 part_idx) const noexcept;
    RegBank reg_bank(u32 part_idx) const noexcept;
};

ValParts val_parts(IRValueRef val) const noexcept;
```
This functions should return an object that provides information about how a value is divided into parts.
Each part corresponds to a register the value should be stored in.
For each value, you need to provide the number of parts and for each part its size in bytes and register bank.
The [RegBank](@ref RegBank) type is defined in [RegisterFile.hpp](@ref tpde/include/tpde/RegisterFile.hpp). The possible
banks are defined in the PlatformConfigs provided by the architecture-specific Compiler classes.

#### val_ref_special
```cpp
using ValRefSpecial = /* optional special ValRef type */;

std::optional<ValRefSpecial> val_ref_special(IRValueRef val) noexcept;
```
With this function you have the possibility to return custom `ValueRef`s when you or the compiler calls `value_ref` with `val`.
This is useful to implement constants that should not be materialized in registers if not needed or otherwise do not need an assignment.
There is a default `ValRefSpecial` defined in `CompilerBase` but you can optionally override this typedef if you need to store additional information
for when individual parts of that value are accessed.

More about this later when explaining how to implement constants.

> [!warning]
> The `ValRefSpecial` type is used in a union so it must be a standard-layout struct and have `u8 mode` as its first member which has to have a value
> greater than 3.

#### val_part_ref_special
```cpp
ValuePart val_part_ref_special(ValRefSpecial& val_ref_special, u32 part_idx) noexcept;
```
When you return a `ValRefSpecial` in `val_ref_special` and a part of it is accessed, this function is called and you need to provide a `ValuePart`
for it.

More details about what `ValuePart`s can be found below.

#### define_func_idx
```cpp
void define_func_idx(IRFuncRef func, u32 idx) noexcept;
```
Internally, the compiler numbers functions and this callback gives you the chance to save that number if you need it later, e.g. to access the created
function symbols. This function is optional to implement.

#### setup_var_ref_assignments
```cpp
void setup_var_ref_assignments() noexcept;
```
This function is called when you set `DEFAULT_VAR_REF_HANDLING` to `false` in the config,
asking you to setup assignments for values which are marked to be "variable references". The details of this are explained further down below.

#### load_address_of_var_reference
```cpp
void load_address_of_var_reference(AsmReg dst, AssignmentPartRef ap) noexcept;
```
This function is called whenever the `ValuePartRef` of a value which is marked as being a custom variable reference is needed in a register.
This function is only needed when `DEFAULT_VAR_REF_HANDLING` is `false`.
The details of this mechanism are explained further down below.

#### cur_cc_assigner
```cpp
CCAssigner* cur_cc_assigner() noexcept;
```
This function provides a calling convention assigner for the current function. It is optional to implement but can be used
if different calling conventions than the default C calling convention should be used. Note that the pointer returned
is non-owning so you need to allocate storage that is live throughout the function for the assigner.

#### compile_inst
```cpp
bool compile_inst(IRInstRef inst, InstRange remaining_instructions) noexcept;
```
This function is called when the compiler iterates through blocks in the code generation pass and asks you to generate machine code for an instruction
and update the value assignments for the result values of the instruction.
The return value indicates whether compilation of the instruction failed.

How to implement this function is explained in the next section

## How to compile instructions

### very simple
- simple example: instruction without any operands, no result, no control flow
- each architecture-compiler provides `ASM*` macros to append instructions to the current function
- use assembler libraries fadec and disarm for encoding, instruction name for the macro derived from the naming in the libraries
```cpp
bool compile_first_inst() {
    ASM(AND32ri, AsmReg::AX, 0xFFFFFFFF);
    return true;
}
```
- registers given to the encoders do not encode their size, this is given in the instruction name

### defining results
- suppose we have instruction that simply returns a zero value
- have to get result [ValueRef](@ref ValueRef)
- [ValueRef](@ref ValueRef) is a RAII wrapper that manages the refcount for a value
- analyzer counts definition and each use (so you will have to retrieve a `ValueRef` for each definition or use as an operand)
- on destruction `ValueRef` will decrement refcount and free associated resources if the value becomes unused
- allows some manual refcounting if necessary (e.g. for fusing)
- when *defining* a value we use the [result_ref](@ref CompilerBase::result_ref) function to get the [ValueRef](@ref ValueRef)
- for each part that we use we can get a [ValuePartRef](@ref ValuePartRef) using [part](@ref ValueRef::part)
- allows to allocate a register when defining a value using [alloc_reg](@ref ValuePartRef::alloc_reg)
> [!note]
> this *locks* the register for the `ValuePartRef` meaning it cannot be evicted until the `ValuePartRef` is destructed or you call [unlock](@ref ValuePartRef::unlock)
- when value defined, you have to call [set_modified](@ref ValuePartRef::set_modified) to tell the framework that the value needs to be spilled if the register is reused
```cpp
bool compile_zero_inst(IRInstRef inst) {
    IRValueRef res_val = /* IR-specific way to get result for instruction */;
    ValueRef res_ref = this->result_ref(res_val);
    ValuePartRef res_part = res_ref.part(0); // in this case, we assume the value is <= 64 bit
    AsmReg res_reg = res_part.alloc_reg(); // allocate a register for the part
    ASM(XOR32rr, res_reg, res_reg); // zero it
    res_part.set_modified(); // tell the framework the part is modified
    return true;
}
```

- if value has multiple parts to zero, e.g. 128 bit int on 64 bit arch, simply get multiple parts
```cpp
bool compile_zero_inst128(IRInstRef inst) {
    IRValueRef res_val = /* IR-specific way to get result for instruction */;
    ValueRef res_ref = this->result_ref(res_val);
    // clear the lower 64 bits
    ValuePartRef res_part = res_ref.part(0);
    AsmReg res_reg = res_part.alloc_reg();
    ASM(XOR32rr, res_reg, res_reg);
    res_part.set_modified();
    // clear the upper 64 bits
    res_part = res_ref.part(1);
    res_reg = res_part.alloc_reg();
    ASM(XOR32rr, res_reg, res_reg);
    res_part.set_modified();
    return true;
}
```

- in case you only need part 0, you can get the `ValueRef` and `ValuePartRef` in an `std::pair` using [result_ref_single](@ref CompilerBase::result_ref_single)
```cpp
auto [res_ref, res_part] = this->result_ref_single(res_val);
```

### using operands
- assume we have a 64 bit add
- need to get lhs and rhs operand
- similar to result: get [ValueRef](@ref ValueRef) and then the necessary [ValuePartRef](@ref ValuePartRef)s
- when using as operand, use [val_ref](@ref CompilerBase::val_ref) function, or [val_ref_single](@ref CompilerBase::val_ref_single)
- to get register, use [load_to_reg](@ref ValuePartRef::load_to_reg) which will also lock the register and reload the value from the stack if necessary
```cpp
bool compile_add64(IRInstRef inst) {
    IRValueRef lhs_val = /* IR-specific way to get operand */;
    IRValueRef rhs_val = /* IR-specific way to get operand */;
    IRValueRef res_val = /* IR-specific way to get result */;
    auto [lhs_ref, lhs_part] = this->val_ref_single(lhs_val);
    auto [rhs_ref, rhs_part] = this->val_ref_single(rhs_val);
    auto [res_ref, res_part] = this->result_ref_single(res_val);

    // load and allocate registers
    AsmReg lhs_reg = lhs_part.load_to_reg();
    AsmReg rhs_reg = rhs_part.load_to_reg();
    AsmReg res_reg = res_part.alloc_reg();

    // encode instruction
    // lhs_reg might be different from res_reg so we have to move first since we cannot override the register of an operand
    ASM(MOV64rr, res_reg, lhs_reg);
    // then do the add
    ASM(ADD64rr, res_reg, rhs_reg);
    res_part.set_modified();
    return true;
}
```

### reusing operand registers
- previous code generation inefficient
- if lhs is not used after the instruction, we could reuse register ("salvage" the register)
- to do this if possible we will use the [into_temporary](@ref ValuePartRef::into_temporary) function
- will give use an owning reference to the register if the value is dead after this instruction or make a copy into a new register
```cpp
bool compile_add64(IRInstRef inst) {
    IRValueRef lhs_val = /* IR-specific way to get operand */;
    IRValueRef rhs_val = /* IR-specific way to get operand */;
    IRValueRef res_val = /* IR-specific way to get result */;
    auto [lhs_ref, lhs_part] = this->val_ref_single(lhs_val);
    auto [rhs_ref, rhs_part] = this->val_ref_single(rhs_val);
    auto [res_ref, res_part] = this->result_ref_single(res_val);

    ValuePartRef tmp_part = lhs_part.into_temporary();

    // load and allocate registers
    AsmReg lhs_reg = tmp_part.cur_reg(); // tmp_part is guaranteed to already own a register
    AsmReg rhs_reg = rhs_part.load_to_reg();
    // do not allocate a register for the result

    // encode instruction
    // lhs_reg contains the value of lhs_part and will also hold the result
    ASM(ADD64rr, lhs_reg, rhs_reg);

    // transfer ownership of the register to res_part (note the std::move here)
    res_part.set_value(std::move(tmp_part));
    // no need to call set_modified, set_value will implicitly do that
    return true;
}
```
- note that in this case we do not need to get the result `ValueRef` before encoding the function and can transform it into a one-liner:
`this->result_ref(res_val).part(0).set_value(std::move(tmp_part))`

- to salvage either the lhs or rhs we can check whether we can actually salvage the register using [can_salvage](@ref ValuePartRef::can_salvage)
- then populate the ValuePartRef using either `lhs_part` or `rhs_part`

### temporary registers
- sometimes temporary registers required for computing result and using `into_temporary` is not sufficient
- allocated and free'd using [ScratchReg](@ref ScratchReg)
- allocate using [alloc](@ref ScratchReg::alloc), [alloc_gp](@ref ScratchReg::alloc_gp) or a specific register using [alloc_specific](@ref ScratchReg::alloc_specific)
- use the `AsmReg` as desired
- register is free'd on destruction of the `ScratchReg` or when calling [reset](@ref ScratchReg::reset)

## Constants using ValRefSpecial
- constants relatively common feature
- they might not have a local index assigned by the adaptor
- might want to handle them just as normal values for code that does not need to care
- every time `val_ref` is called it will call `val_ref_special` in your compiler
- check if the value is a constant and then return a `ValRefSpecial`
- suppose we use the default [ValRefSpecial](@ref CompilerBase::ValRefSpecial) from CompilerBase since we only support 64 bit constants
- then implement `val_ref_special` like this
```cpp
std::optional<ValRefSpecial> val_ref_special(IRValueRef value) noexcept {
    if (/* IR-specific way to check if value is not constant */) {
        return std::nullopt;
    }

    u64 constant = /* IR-specific way to get constant data from value */;
    return ValRefSpecial{.mode = 4, .const_data = constant};
}
```

- since constants are only 64 bit, only part 0 will be accessed for x86-64
- in `val_part_ref_special` we can then return a constant `ValuePartRef` which will handle materialization for us
```cpp
ValuePartRef val_part_ref_special(ValRefSpecial& val_ref, u32 part_idx) noexcept {
    assert(part_idx == 0);
    // ValuePartRef has a constructor that takes 64 bits of constant data, its actual size in bytes and the register bank for the constant
    // in our case the size will be 8 and the constant will be an integer
    return ValuePartRef{val_ref.const_data, 8, Config::GP_BANK};
```
- for constants larger than 64 bits, you will need to allocate them somewhere and then give the `ValuePartRef` a pointer to the constant data (e.g. for vector constants)
- if you want to store more information about constants, you will need to define your own `ValRefSpecial` struct
- LLVM implementation can be used as a reference

### optimizing constant operands
- might want to optimize instruction selection if an operand is a constant since many instructions can take a register or immediate operand
- doable by checking if a part is constant using [is_const](@ref ValuePartRef::is_const)
- then access the constant data using [const_data](@ref ValuePartRef::const_data) which returns a `std::span<u64>`
```cpp
bool compile_add64(IRInstRef inst) {
    IRValueRef lhs_val = /* IR-specific way to get operand */;
    IRValueRef rhs_val = /* IR-specific way to get operand */;
    IRValueRef res_val = /* IR-specific way to get result */;
    auto [lhs_ref, lhs_part] = this->val_ref_single(lhs_val);
    auto [rhs_ref, rhs_part] = this->val_ref_single(rhs_val);
    auto [res_ref, res_part] = this->result_ref_single(res_val);

    ValuePartRef tmp_part = lhs_part.into_temporary();

    // load and allocate registers
    AsmReg lhs_reg = tmp_part.cur_reg(); // tmp_part is guaranteed to already own a register
    if (rhs_part.is_const() // check if part is const
        && i64(rhs_part.const_data()[0]) == i64(i32(rhs_part.const_data()[0])) // check if it can be encoded as a 32 bit immediate in x86-64
        ) {
        // encode using immediate operand
        ASM(ADD64ri, lhs_reg, rhs_part.const_data()[0]);
    } else {
        // encode as before
        AsmReg rhs_reg = rhs_part.load_to_reg();
        ASM(ADD64rr, lhs_reg, rhs_reg);
    }

    res_part.set_value(std::move(tmp_part));
    return true;
}
```

- can also optimize if lhs is const and rhs is not
- before calling `into_temporary`
```cpp
if (lhs_part.is_const() && !rhs_part.is_const()) {
    std::swap(lhs_ref, rhs_ref);
    std::swap(lhs_part, rhs_part);
}
```

## Materializing constants
- sometimes need constant that is not a value for computation in register
- materializing constants might be tedious (AArch64)
- simple helper to do that if you have allocated a `ScratchReg`
```cpp
ScratchReg scratch{this};
AsmReg tmp_reg = scratch.alloc_gp();
u64 constant = /* some constant */;
this->materialize_constant(constant, Config::GP_BANK, /* size_bytes = */ 8, tmp_reg);
// constant is now in tmp_reg
```

## Fusing instructions
- sometimes it is beneficial to fuse adjacent IR instructions, e.g. load with zero extend
- TPDE only supports fusing instructions which have not been compiled yet (forward fusing)
- adaptor needs to implement bookkeeping for which instructions have been fused
- `compile_inst` gets an `InstRange` as a parameter which can be used to iterate over the following instructions
- check for the specific instruction we want to match, then mark it as fused
- example: load i8 with sx
```cpp
bool compile_loadi8(IRInstRef inst, InstRange remaining) {
    IRValueRef ptr_val = /* IR-specific */;
    IRValueRef res_val = /* IR-specific */;

    auto [ptr_ref, ptr_part] = this->val_ref_single(ptr_val)
    AsmReg ptr_reg = ptr_part.load_to_reg();

    if (remaining.from != remaining.to // is there any instruction left?
        && analyzer.liveness_info(u32(this->adaptor->val_local_idx(res_val))).ref_count <= 2 // does the current load only have one user? (definition counts as a use here)
        && /* IR-specific; *remaining.from yields an IRInstRef */ // is the following instruction a sign extension?
        ) {
        res_val = /* IR-specific way to get result value ref from *remaining.from */;
        auto [res_ref, res_part] = this->result_ref_single(res_val);

        AsmReg res_reg = res_part.alloc_reg();
        ASM(MOVSXr64m8, res_reg, FE_MEM(/* base = */ ptr_reg, /* scale = */ 0, /* index_reg = */ FE_NOREG, /* displacement = */ 0));
        res_part.set_modified();
        // mark *remaining.from as fused here
    } else {
        // use regular MOVZX
        res_val = /* IR-specific way to get result value ref from *remaining.from */;
        auto [res_ref, res_part] = this->result_ref_single(res_val);

        AsmReg res_reg = res_part.alloc_reg();
        ASM(MOVZXr32m8, res_reg, FE_MEM(/* base = */ ptr_reg, /* scale = */ 0, /* index_reg = */ FE_NOREG, /* displacement = */ 0));
        res_part.set_modified();
    }
    return true;
}
```

## Return
- use [RetBuilder](@ref CompilerBase::RetBuilder) to assign returned ValueParts or IRValueRefs to their corresponding return registers
- ValueParts may be used if you need to do some processing before returning them, e.g. zero-/sign-extension with non-power-of-two bitwidths
- afterwards, call [RetBuilder::ret](@ref CompilerBase::RetBuilder::ret) to generate the return
- this will internally move the results into the return registers and then call `gen_func_epilog` and `release_regs_after_return`
- if you do not need to return anything, you can skip the RetBuilder and call these two functions yourself

> [!warning]
> you *always* need to call `release_regs_after_return` if you compile an instruction that terminates a basic block and does not branch to another block

```cpp
bool compile_ret(IRInstRef inst) {
    // Create the RetBuilder
    RetBuilder rb{this, *this->cur_cc_assigner()};

    if (/* IR-specific way to check if return should return a value */) {
        IRValueRef ret_val = /* IR-specific way to get return value */;
        // add the returned value to the returns
        rb.add(ret_val);
    }

    // generate the return
    rb.ret();
    return true;
}
```

## Branches
There are high-level functions to generate branches/switches and low-level primitives for more special use cases.

> [!warning]
> Never emit a branch to a block other than using `generate_branch_to_block` or the high-level functions that do this internally.

### High-Level Functions
- These functions can only be called at the end of a block. No value parts must be locked when these functions are called.
- Unconditional branches can be generated with [generate_uncond_branch](@ref tpde::CompilerBase::generate_uncond_branch).
- Conditional branches can be generated with [generate_cond_branch](@ref tpde::CompilerBase::generate_cond_branch). The possible branching conditions are architecture-specific. Typically the branch condition involves reading the status flags register, this must be set beforehand. The function might invert the branch condition to implicitly fall through to the next block.
- Switches over integers can be generated with [generate_switch](@ref tpde::CompilerBase::generate_switch). The switch operand must be in a single register, only switches up to 64 bits are supported.

```cpp
bool compile_br(IRInstRef inst) {
    IRBlockRef target = /* IR-specific way to get target block ref */;
    this->generate_uncond_branch(target);
    return true;
}
bool compile_condbr(IRInstRef inst) {
    IRValueRef cond_val = /* IR-specific way to get condition */;
    IRBlockRef true_target = /* IR-specific way to get block ref */;
    IRBlockRef false_target = /* IR-specific way to get block ref */;

    // Note: the condition must not be locked and must be ref-counted when
    // generate_cond_branch is called.
    {
        auto [cond_ref, cond_part] = this->val_ref_single(cond_val);
        ASM(CMP32ri, cond_part.load_to_reg(), 0);
    }

    this->generate_cond_branch(Jump::jne, true_target, false_target);
    return true;
}
```

### Low-Level Branching Functions
- branching mostly handled by the compiler
- use `generate_branch_to_block` from the architecture compiler to emit an (un)conditional branch, arguments are architecture-specific
- before calling it, need to call [spill_before_branch](@ref tpde::CompilerBase::spill_before_branch) so that values which live across blocks
  can be spilled and [begin_branch_region](@ref tpde::CompilerBase::begin_branch_region) which asserts that no value assignments are changed during branching
- after all branches are generated, call [end_branch_region](@ref tpde::CompilerBase::end_branch_region) to end code generation for branches and [release_spilled_regs](@ref tpde::CompilerBase::release_spilled_regs) so the register allocator can free values which can no longer be assumed to be in registers after the branch

#### Unconditional branch
- need to get `IRBlockRef` to jump to
- straightforward
```cpp
bool compile_br(IRInstRef inst) {
    IRBlockRef target = /* IR-specific way to get target block ref */;

    const auto spilled = this->spill_before_branch();
    this->begin_branch_region();

    this->generate_branch_to_block(Jump::jmp, target, /* needs_split = */ false, /* last_inst = */ true);

    this->end_branch_region();
    this->release_spilled_regs(spilled);

    return true;
}
```
- split only relevant to conditional branches

#### Conditional branch
- a bit more involved but can also be simple
- suppose jump to false block only when lower 32 bit 0, otherwise true block
- need to ask framework whether branch to block needs to be split
```cpp
bool compile_condbr(IRInstRef inst) {
    IRValueRef cond_val = /* IR-specific way to get condition */;
    IRBlockRef true_target = /* IR-specific way to get block ref */;
    IRBlockRef false_target = /* IR-specific way to get block ref */;

    {
        auto [cond_ref, cond_part] = this->val_ref_single(cond_val);
        ASM(CMP32ri, cond_part.load_to_reg(), 0);
    }

    const bool true_needs_split = this->branch_needs_split(true_target);
    
    // let the framework spill
    const auto spilled = this->spill_before_branch();
    this->begin_branch_region();

    // actually generate the branches
    this->generate_branch_to_block(Jump::jne, true_target, true_needs_split, /* last_inst = */ false);
    // unconditional branch since the condition must be false at this point
    // note that the last branch in a block never needs to be split
    this->generate_branch_to_block(Jump::jmp, false_target, /* needs_split = */ false, /* last_inst = */ true);

    // framework register handling
    this->end_branch_region();
    this->release_spilled_regs(spilled);
    return true;
}
```

- want to optimize this, since we want to avoid branch to block that comes directly after
- a bit more involved; it is typically preferable to call @ref tpde::CompilerBase::generate_cond_branch instead.
```cpp
bool compile_condbr(IRInstRef inst) {
    IRValueRef cond_val = /* IR-specific way to get condition */;
    IRBlockRef true_target = /* IR-specific way to get block ref */;
    IRBlockRef false_target = /* IR-specific way to get block ref */;

    {
        auto [cond_ref, cond_part] = this->val_ref_single(cond_val);
        ASM(CMP32ri, cond_part.load_to_reg(), 0);
    }

    const bool true_needs_split = this->branch_needs_split(true_target);
    const bool false_needs_split = this->branch_needs_split(false_target);
    const IRBlockRef next_block = this->analyzer.block_ref(this->next_block());
    
    // let the framework spill
    const auto spilled = this->spill_before_branch();
    this->begin_branch_region();

    if (next_block == true_target || (next_block != false_target && true_needs_split)) {
        // if the following block is the true target or if we have to always emit a branch but a branch to the true block
        // is heavy (i.e. needs to be split) then we want to first jump to the false block
        this->generate_branch_to_block(this->invert_jump(Jump::jne), false_target, false_needs_split, /* last_inst = */ false);
        // if the next block is the true_target, then the jump will not be emitted
        this->generate_branch_to_block(Jump::jmp, true_target, /* needs_split = */ false, /* last_inst = */ true);
    } else {
        // try to elide the branch to the false_target
        this->generate_branch_to_block(Jump::jne, true_target, true_needs_split, /* last_inst = */ false);
        this->generate_branch_to_block(Jump::jmp, false_target, /* needs_split = */ false, /* last_inst = */ true);
    } 

    // framework register handling
    this->end_branch_region();
    this->release_spilled_regs(spilled);
    return true;
}
```

## Emitting function calls
- calls emitted using [CallBuilder](@ref CompilerBase::CallBuilder) helper or `generate_call` helper function in architecture compiler
- two types of target:
  - direct call to symbol
  - indirect call to ptr in ValuePartRef

### generate_call
- simpler API
- arguments given by their IRValueRef + flags for sign/zero-extension or passed as byval
- results returned in [ValueRef](@ref ValueRef)
- supports calling vararg functions
- will always use the default C calling convention

```cpp
bool compile_call_direct(IRInstRef inst) {
    IRValueRef res_val = /* IR-specific way to get result value */;
    u32 target_idx = /* implementation-specific way to get index passed to define_func_idx for the target function */;
    SymRef target_sym = this->func_syms[target_idx];

    tpde::SmallVector<CallArg, 4> args{};
    for (IRValueRef call_arg : /* IR-specific */) {
        // no extensions, no byval arguments
        args.push_back(CallArg{call_arg});
    }

    // only one result value
    ValueRef res_ref = this->result_ref(res_val);

    this->generate_call(target_sym, args, &res_ref, /* variable_args = */ false);
    return true;
}
```

- other ways to get symbols, you can create your own, e.g. LLVM back-end does this

### CallBuilder
- more complex but more flexible API
- defined in architecture compiler
- takes a reference to a [CCAssigner](@ref CompilerBase::CCAssigner) to allocate argument and result register
- vararg configured in CCAssigner
- if a custom CCAssigner is used (e.g. `tpde::x64::CCAssignerSysV`), storage for it needs to be allocated and held live as long as the CallBuilder is live
- for each argument call [add_arg](@ref CompilerBase::CallBuilder::add_arg) with a [CallArg](@ref CompilerBase::CallArg) (IRValueRef + flags) or with a ValuePart and a [CCAssignment](@ref tpde::CCAssignment) which contains more fine-grained information
- then use [call](@ref CompilerBase::CallBuilder::call) to generate the call
- use [add_ret](@ref CompilerBase::CallBuilder::add_ret) to collect the result values


```cpp
bool compile_call_direct(IRInstRef inst) {
    IRValueRef res_val = /* IR-specific way to get result value */;
    u32 target_idx = /* implementation-specific way to get index passed to define_func_idx for the target function */;
    SymRef target_sym = this->func_syms[target_idx];

    // use the current calling convention for the call
    CallBuilder cb{*this, *this->cur_cc_assigner()};

    for (IRValueRef call_arg : /* IR-specific */) {
        // not needed here, we could just call
        // cb.add_arg(CallArg{call_arg});

        ValueRef vr = this->val_ref(call_arg);
        // assume that value has only one part
        CCAssignment cca;
        cb.add_arg(vr.part(0), cca);
    }

    // generate the call
    cb.call(target_sym);

    // only one result value
    ValueRef res_ref = this->result_ref(res_val);
    
    // we could also just call
    // cb.add_ret(res_ref)
    CCAssignment cca;
    // assume only one part
    cb.add_ret(res_ref.part(0));

    return true;
}
```

## Manual stack allocations
- manual allocation of stack slots sometimes needed, e.g. when an instruction needs too many temporaries or manual handling of static allocations
- allocated using [allocate_stack_slot](@ref CompilerBase::allocate_stack_slot), free'd using [free_stack_slot](@ref CompilerBase::free_stack_slot)
> [!note]
> only free when all blocks in which the stack slot could be live have been compiled

```cpp
ScratchReg scratch{this};
AsmReg reg = scratch.alloc_gp();
// calculate some value in reg...

// allocate slot and spill
i32 slot = this->allocate_stack_slot(/* size_bytes = */ 8);
this->spill_reg(reg, slot, /* size_bytes = */ 8); // in architecture-compiler

// use reg for something else

// reload the value from the stack
this->load_from_stack(reg, slot, /* size_bytes = */ 8, /* sign_extend = */ false); // in architecture compiler
// use original value of reg...

// free the stack slot
this->free_stack_slot(slot, /* size_bytes = */ 8);
```

## Custom Variable Ref Handling
- ValueRefs can be marked as being "variable references"
- i.e. this means that these values are pointers that are easy to calculate on-demand and are not spilled when their registers are evicted
- by default, ValueRefs for static stack allocations are classified as variable references
- these also have the `stack_variable` bit set in the ValueAssignment
- compiler has option to allow user to add custom kinds of variable references, e.g. global value pointers
- set `DEFAULT_VAR_REF_HANDLING` to `false`
- then your compiler will need to implement `setup_var_ref_assignments` and `load_address_of_var_reference`:
  - `setup_var_ref_assignments` can initialize the assignments of values which need to be valid at the beginning of the function using [init_variable_ref](@ref CompilerBase::init_variable_ref)
  - `load_address_of_var_reference` will need to calculate the address of the value the variable reference points to and load it into a register
- user can use `variable_ref_data` in the [AssignmentPartRef](@ref tpde::AssignmentPartRef) of the first part to store custom information about the value, e.g. an index into a custom data structure
- you will need to handle access to a custom variable ref assignment in `val_ref_special`/`val_part_ref_special` and initialize it
  using [init_variable_ref](@ref CompilerBase::init_variable_ref) if necessary
- this can then return a normal [ValuePartRef](@ref CompilerBase::ValuePartRef) referencing this assignment
- An implementation of this can be found in TPDE-LLVM

## Architecture-specific

### Calling Conventions
- each architecture defines a set of CCAssigners which handle register/stack allocation for argument and return registers
  for a calling convention
- your compiler may implement the `cur_cc_assigner` function that returns the assigner for the current function
- `generate_call` always uses the default C calling convention
- CallBuilders take a CCAssigner as a constructor argument to specify the calling convention to use
- currently, only SysV/AAPCS is supported

## Assembler
- manages data structures for code and data sections, their relocations, symbols and Labels (i.e. function-local symbols)
- handles generating an object file and unwind information
- configured using the compiler config, instance stored in CompilerBase
- currently only x86-64 and AArch64 Linux ELF supported
- unwind info is synchronous
- supports generating C++ exception information

### Sections
- various sections for code or data created
- accessed using `SecRef`s
- current code section stored in Assembler as [current_section](@ref AssemblerElf::current_section)
- other sections accessed using `get_{data,bss,tdata,tbss,structor,eh_frame}_section`

### Symbols
- SymRef used to reference symbols
- various functions to create and define symbols

#### SymBinding
- three levels to define linkage of symbols, reflecting ELF semantics:
  - local
  - weak
  - global

#### sym_add_undef
```cpp
SymRef sym_add_undef(std::string_view name, SymBinding binding) noexcept;
```
Adds an undefined symbol, possibly with a name.
Can be defined later using [sym_def](@ref AssemblerElf::sym_def).

#### sym_predef_*
```cpp
SymRef sym_predef_func(std::string_view name, SymBinding binding) noexcept;
SymRef sym_predef_data(std::string_view name, SymBinding binding) noexcept;
SymRef sym_predef_tls(std::string_view name, SymBinding binding) noexcept;
```
Predefines a symbol for a function, data or TLS object respectively. Must be defined later using [sym_def](@ref AssemblerElf::sym_def)
for function or TLS objects and using [sym_def_predef_data](@ref AssemblerElf::sym_def_predef_data) for data objects.

#### sym_def_predef_data
```cpp
void sym_def_predef_data(SecRef sec, SymRef sym, std::span<const u8> data, u32 align, u32 *off) noexcept;
```
Defines a previously declared data object in the section `sec` with the specified data and alignment.
If `off` is not a `nullptr`, the offset of the data in the section is written to the `u32` pointed to by `off`.

#### sym_def_data
```cpp
SymRef sym_def_data(SecRef sec, std::string_view name, std::span<const u8> data, u32 align, SymBinding binding, u32 *off = nullptr) noexcept;
```
Shortcut for [sym_predef_data](@ref AssemblerElf::sym_predef_data) followed by [sym_def_predef_data](@ref AssemblerElf::sym_def_predef_data)

### Relocations
- assembler can generate relocations
- useful if constant data is needed or globals are used
- usually target-dependent, you need to know what to generate
- example: GOT address load
```cpp
void load_address_of_global(SymRef global_sym, AsmReg dst) {
    // emit a mov from the GOT
    // the following macro invocation will generate a `mov dst, [rip - 1]` to force a 32 bit displacement immediate to be generated
    ASM(MOV64rm, dst, FE_MEM(/* base = */ FE_IP, /* scale = */ 0, /* idx_reg = */ FE_NOREG, /* disp = */ -1));

    // the immediate operand of the MOV64rm is 4 bytes and comes at the end of the instruction
    // since the offset is calculated by the CPU from the end of the instruction, we need an addend of -4
    this->assembler.reloc_text(global_sym, /* type = */ R_X86_64_GOTPCREL, this->assembler.text_cur_off() - 4, /* addend = */ -4);
}
```

#### reloc_sec
```cpp
void reloc_sec(SecRef sec, SymRef sym, u32 type, u64 offset, i64 addend) noexcept;
```
Adds a relocation of `type` (architecture-/platform-specific) in section `sec` at `offset` to the specified symbol `sym`
with an addend.

#### reloc_pc32
```cpp
void reloc_pc32(SecRef sec, SymRef sym, u64 offset, i64 addend) noexcept;
```
Adds a 32-bit pc-relative relocation in section `sec` at `offset` to the specified symbol `sym` with an addend in an
architecture-independent matter.

#### reloc_abs
```cpp
void reloc_abs(SecRef sec, SymRef sym, u64 offset, i64 addend) noexcept;
```
Adds a 64-bit absolute relocation, i.e. writes the address of `sym`, in section `sec` at `offset` to the specified symbol `sym`
with an addend in an architecture-independent matter.

#### reloc_text
```cpp
void reloc_text(SymRef sym, u32 type, u64 offset, i64 addend = 0) noexcept;
```
Adds a reloctation to `sym` in the current text section at `offset` of `type` (architecture-/platform-specific) with an optional addend.

### Labels
- markers for code in current function, labels are cleared at the end of a function
- used for `generate_raw_jump` in architecture compilers to encode control flow when compiling a single instruction
- create with [label_create](@ref FunctionWriter::label_create)
- place with [label_place](@ref CompilerBase::label_place)
- all control flow within an instruction must converge at the end of it, i.e. no branches to blocks inside control flow


```cpp
void compile_highest_bit_set(IRInstRef inst) noexcept {
    // get input and output values for instruction
    AsmReg in_reg = /* ... */;
    AsmReg out_reg = /* ... */;
    ScratchReg tmp{this};
    AsmReg tmp_reg = tmp.alloc_gp();
    
    // very inefficient implementation...

    // create labels for jump targets
    Label fin = this->text_writer.label_create();
    Label cont = this->text_writer.label_create();

    ASM(XOR32rr, out_reg, out_reg);
    ASM(DEC64r, out_reg);
    ASM(MOV64rr, tmp_reg, in_reg);

    // place continuation label
    this->label_place(cont);

    // check if tmp is zero
    ASM(CMP64ri, tmp_reg, 0);

    // break out of the loop if it is
    this->generate_raw_jump(Jump::jz, fin);
    
    // increment counter and shift tmp
    ASM(INC64r, out_reg);
    ASM(SHR64ri, tmp_reg, 1);

    // jump to beginning of the loop
    this->generate_raw_jump(Jump::jmp, cont);

    // place the label for the end of the loop
    this->label_place(fin);

    // no more code generation necessary

    // set the result register for the result value...
}
```

> [!warning]
> Don't emit code that might cause registers to be spilled when in conditional control-flow

#### Jump Tables
- for switch tables it is beneficial to generate jump tables which encode offsets to labels
> [!note]
> you cannot directly jump to blocks using an offset since the register allocator might need
> to do PHI moves at the edge. You first have to jump to a label that then branches to the target block
- Assembler has helper function to manually add an unresolved offset to a label

```cpp
void write_jump_table(Label jump_table, std::span<Label> cases) {
    SecRef text_ref = this->text_writer.get_sec_ref();

    // offsets are 32 bit in size
    this->text_writer.align(4);

    // we want the jump table to be continuous
    this->text_writer.ensure_space(4 + 4 * labels.size());

    this->label_place(jump_table);

    // offsets to case labels should be relative to the start of the table
    u32 table_off = this->text_writer.offset();
    for (Label case_label : cases) {
        // label_ref can only be used for labels which aren't placed yet
        // depending on how you use this, the labels might always be unplaced
        if (this->text_writer.label_is_pending(case_label)) {
            this->text_writer.label_ref(
                case_label,
                this->text_writer.offset(),
                tpde::LabelFixupKind::X64_JUMP_TABLE // for x64
            );
            // write the table_offset, this will be used to calculate the actual offset when the label is placed
            this->text_writer.write<u32>(table_off);
        } else {
            // write the correct offset
            u32 label_off = this->text_writer.label_offset(case_label);
            this->text_writer.write<i32>((i32)label_off - (i32)table_off);
        }
    }
}
```

## Compiling and Generating Object File
- When compilation is finished (CompilerBase::compile returns), Assembler can be used to create object or write to memory

### Generate object
- [build_object_file](@ref AssemblerElf::build_object_file) writes out a finished ELF object file to a vector

```cpp
void compile_elf() {
    // build compiler
    if (!compiler->compile()) {
        return;
    }

    std::vector<u8> obj = compiler->assembler.build_object_file();
    // write to disk, pass to mapper, etc...
}
```

### Mapping to memory
- For ELF Linux x64/AArch64 there is a helper class [ElfMapper](@ref ElfMapper) which maps the compiled sections directly into memory
- Need to provide a function to resolve undefined symbols
- Get symbol address using [get_sym_addr](@ref ElfMapper::get_sym_addr)
- On destruction, automatically free's the memory and deregisters unwind info
> [!note]
> can be reused after reset

```cpp
void compile_and_map() {
    // build compiler
    if (!compiler->compile()) {
        return;
    }

    tpde::ElfMapper mapper{};
    if (!mapper.map(compiler->assembler, [](std::string_view sym_name) -> void* { /* resolve symbol */ })) {
        return;
    }

    // code mapped to memory

    // get address of first function compiled
    void* fun_addr = mapper.get_sym_addr(compiler->func_syms[0]);
    // and call it
    static_cast<void(*)()>(fun_addr)();

    // free memory manually
    mapper.reset();
}
```

<div class="section_buttons">
 
| Previous          |                              Next |
|:------------------|----------------------------------:|
| [IRAdaptor Reference](@ref tpde-adaptor) | [EncodeGen Reference](@ref tpde-encodegen) |
 
</div>

