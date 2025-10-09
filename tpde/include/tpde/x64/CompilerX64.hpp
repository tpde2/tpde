// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include "tpde/AssemblerElf.hpp"
#include "tpde/AssignmentPartRef.hpp"
#include "tpde/CompilerBase.hpp"
#include "tpde/base.hpp"
#include "tpde/x64/FunctionWriterX64.hpp"

#include <bit>

// Helper macros for assembling in the compiler
#if defined(ASM) || defined(ASMF) || defined(ASMNC) || defined(ASME)
  #error Got definition for ASM macros from somewhere else. Maybe you included compilers for multiple architectures?
#endif

// Use helper, parameters might call ASM themselves => evaluate text_cur_ptr
// after the arguments.
#define ASM_FULL(compiler, reserve, op, ...)                                   \
  ((compiler)->asm_helper(fe64_##op).encode(reserve, __VA_ARGS__))

#define ASM(op, ...) ASM_FULL(this, 16, op, 0 __VA_OPT__(, ) __VA_ARGS__)
#define ASMC(compiler, op, ...)                                                \
  ASM_FULL(compiler, 16, op, 0 __VA_OPT__(, ) __VA_ARGS__)
#define ASMF(op, flag, ...)                                                    \
  ASM_FULL(this, 16, op, flag __VA_OPT__(, ) __VA_ARGS__)
#define ASMNCF(op, flag, ...)                                                  \
  ASM_FULL(this, 0, op, flag __VA_OPT__(, ) __VA_ARGS__)
#define ASMNC(op, ...) ASM_FULL(this, 0, op, 0 __VA_OPT__(, ) __VA_ARGS__)

namespace tpde::x64 {

struct AsmReg : Reg {
  enum REG : u8 {
    AX = 0,
    CX,
    DX,
    BX,
    SP,
    BP,
    SI,
    DI,
    R8,
    R9,
    R10,
    R11,
    R12,
    R13,
    R14,
    R15,

    XMM0 = 32,
    XMM1,
    XMM2,
    XMM3,
    XMM4,
    XMM5,
    XMM6,
    XMM7,
    XMM8,
    XMM9,
    XMM10,
    XMM11,
    XMM12,
    XMM13,
    XMM14,
    XMM15,
    // TODO(ts): optional support for AVX registers with compiler flag
  };

  constexpr explicit AsmReg() noexcept : Reg((u8)0xFF) {}

  constexpr AsmReg(const REG id) noexcept : Reg((u8)id) {}

  constexpr AsmReg(const Reg base) noexcept : Reg(base) {}

  constexpr explicit AsmReg(const u8 id) noexcept : Reg(id) {
    assert(id <= R15 || (id >= XMM0 && id <= XMM15));
  }

  constexpr explicit AsmReg(const u64 id) noexcept : Reg(id) {
    assert(id <= R15 || (id >= XMM0 && id <= XMM15));
  }

  constexpr operator FeRegGP() const noexcept {
    assert(reg_id <= R15);
    return FeRegGP{reg_id};
  }

  operator FeRegGPLH() const noexcept {
    assert(reg_id <= R15);
    return FeRegGP{reg_id};
  }

  constexpr operator FeRegXMM() const noexcept {
    assert(reg_id >= XMM0 && reg_id <= XMM15);
    return FeRegXMM{static_cast<u8>(reg_id & 0x1F)};
  }
};

constexpr static u64
    create_bitmask(const std::initializer_list<AsmReg::REG> regs) {
  u64 set = 0;
  for (const auto reg : regs) {
    set |= 1ull << reg;
  }
  return set;
}

template <size_t N>
constexpr static u64 create_bitmask(const std::array<AsmReg, N> regs) {
  u64 set = 0;
  for (const auto reg : regs) {
    set |= 1ull << reg.id();
  }
  return set;
}

/// x86-64 System V calling convention.
class CCAssignerSysV : public CCAssigner {
public:
  static constexpr CCInfo Info{
      .allocatable_regs =
          0xFFFF'0000'FFFF & ~create_bitmask({AsmReg::BP, AsmReg::SP}),
      .callee_saved_regs = create_bitmask({
          AsmReg::BX,
          AsmReg::R12,
          AsmReg::R13,
          AsmReg::R14,
          AsmReg::R15,
      }),
      .arg_regs = create_bitmask({
          AsmReg::DI,
          AsmReg::SI,
          AsmReg::DX,
          AsmReg::CX,
          AsmReg::R8,
          AsmReg::R9,
          AsmReg::XMM0,
          AsmReg::XMM1,
          AsmReg::XMM2,
          AsmReg::XMM3,
          AsmReg::XMM4,
          AsmReg::XMM5,
          AsmReg::XMM6,
          AsmReg::XMM7,
      }),
      .red_zone_size = 128,
  };

private:
  u32 gp_cnt = 0, xmm_cnt = 0, stack = 0;
  // The next N assignments must go to the stack.
  unsigned must_assign_stack = 0;
  bool vararg;
  u32 ret_gp_cnt = 0, ret_xmm_cnt = 0;

public:
  CCAssignerSysV(bool vararg = false) noexcept
      : CCAssigner(Info), vararg(vararg) {}

  void reset() noexcept override {
    gp_cnt = xmm_cnt = stack = 0;
    must_assign_stack = 0;
    vararg = false;
    ret_gp_cnt = ret_xmm_cnt = 0;
  }

  void assign_arg(CCAssignment &arg) noexcept override {
    if (arg.byval) {
      stack = util::align_up(stack, arg.align < 8 ? 8 : arg.align);
      arg.stack_off = stack;
      stack += arg.size;
      return;
    }

    if (arg.bank == RegBank{0}) {
      static constexpr std::array<AsmReg, 6> gp_arg_regs{
          AsmReg::DI,
          AsmReg::SI,
          AsmReg::DX,
          AsmReg::CX,
          AsmReg::R8,
          AsmReg::R9,
      };
      if (!must_assign_stack && gp_cnt + arg.consecutive < gp_arg_regs.size()) {
        arg.reg = gp_arg_regs[gp_cnt];
        gp_cnt += 1;
      } else {
        // Next N arguments must also be assigned to the stack
        // Increment by one, the value is immediately decremented below.
        must_assign_stack = arg.consecutive + 1;
        stack = util::align_up(stack, arg.align < 8 ? 8 : arg.align);
        arg.stack_off = stack;
        stack += 8;
      }
    } else {
      if (!must_assign_stack && xmm_cnt < 8) {
        arg.reg = Reg{AsmReg::XMM0 + xmm_cnt};
        xmm_cnt += 1;
      } else {
        // Next N arguments must also be assigned to the stack
        // Increment by one, the value is immediately decremented below.
        must_assign_stack = arg.consecutive + 1;
        u32 size = util::align_up(arg.size, 8);
        stack = util::align_up(stack, size);
        arg.stack_off = stack;
        stack += size;
      }
    }

    if (must_assign_stack > 0) {
      must_assign_stack -= 1;
    }
  }

  u32 get_stack_size() noexcept override { return stack; }

  bool is_vararg() const noexcept override { return vararg; }

  void assign_ret(CCAssignment &arg) noexcept override {
    assert(!arg.byval && !arg.sret);
    if (arg.bank == RegBank{0}) {
      if (ret_gp_cnt + arg.consecutive < 2) {
        arg.reg = Reg{ret_gp_cnt == 0 ? AsmReg::AX : AsmReg::DX};
        ret_gp_cnt += 1;
      } else {
        assert(false);
      }
    } else {
      if (ret_xmm_cnt + arg.consecutive < 2) {
        arg.reg = Reg{ret_xmm_cnt == 0 ? AsmReg::XMM0 : AsmReg::XMM1};
        ret_xmm_cnt += 1;
      } else {
        assert(false);
      }
    }
  }
};

struct PlatformConfig : CompilerConfigDefault {
  using Assembler = AssemblerElfX64;
  using AsmReg = tpde::x64::AsmReg;
  using DefaultCCAssigner = CCAssignerSysV;
  using FunctionWriter = FunctionWriterX64;

  static constexpr RegBank GP_BANK{0};
  static constexpr RegBank FP_BANK{1};
  static constexpr bool FRAME_INDEXING_NEGATIVE = true;
  static constexpr u32 PLATFORM_POINTER_SIZE = 8;
  static constexpr u32 NUM_BANKS = 2;
};

namespace concepts {
template <typename T, typename Config>
concept Compiler = tpde::Compiler<T, Config> && requires(T a) {
  {
    a.arg_is_int128(std::declval<typename T::IRValueRef>())
  } -> std::convertible_to<bool>;

  {
    a.arg_allow_split_reg_stack_passing(std::declval<typename T::IRValueRef>())
  } -> std::convertible_to<bool>;
};
} // namespace concepts

/// Compiler mixin for targeting x86-64.
template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy =
              CompilerBase,
          typename Config = PlatformConfig>
struct CompilerX64 : BaseTy<Adaptor, Derived, Config> {
  using Base = BaseTy<Adaptor, Derived, Config>;

  using IRValueRef = typename Base::IRValueRef;
  using IRBlockRef = typename Base::IRBlockRef;
  using IRFuncRef = typename Base::IRFuncRef;

  using ScratchReg = typename Base::ScratchReg;
  using ValuePartRef = typename Base::ValuePartRef;
  using ValuePart = typename Base::ValuePart;
  using GenericValuePart = typename Base::GenericValuePart;

  using Assembler = typename PlatformConfig::Assembler;
  using RegisterFile = typename Base::RegisterFile;

  using CallArg = typename Base::CallArg;

  using Base::derived;


  // TODO(ts): make this dependent on the number of callee-saved regs of the
  // current function or if there is a call in the function?
  static constexpr u32 NUM_FIXED_ASSIGNMENTS[PlatformConfig::NUM_BANKS] = {5,
                                                                           6};

  enum CPU_FEATURES : u32 {
    CPU_BASELINE = 0, // x86-64-v1
    CPU_CMPXCHG16B = (1 << 0),
    CPU_POPCNT = (1 << 1),
    CPU_SSE3 = (1 << 2),
    CPU_SSSE3 = (1 << 3),
    CPU_SSE4_1 = (1 << 4),
    CPU_SSE4_2 = (1 << 5),
    CPU_AVX = (1 << 6),
    CPU_AVX2 = (1 << 7),
    CPU_BMI1 = (1 << 8),
    CPU_BMI2 = (1 << 9),
    CPU_F16C = (1 << 10),
    CPU_FMA = (1 << 11),
    CPU_LZCNT = (1 << 12),
    CPU_MOVBE = (1 << 13),
    CPU_AVX512F = (1 << 14),
    CPU_AVX512BW = (1 << 15),
    CPU_AVX512CD = (1 << 16),
    CPU_AVX512DQ = (1 << 17),
    CPU_AVX512VL = (1 << 18),

    CPU_V2 = CPU_BASELINE | CPU_CMPXCHG16B | CPU_POPCNT | CPU_SSE3 | CPU_SSSE3 |
             CPU_SSE4_1 | CPU_SSE4_2,
    CPU_V3 = CPU_V2 | CPU_AVX | CPU_AVX2 | CPU_BMI1 | CPU_BMI2 | CPU_F16C |
             CPU_FMA | CPU_LZCNT | CPU_MOVBE,
    CPU_V4 = CPU_V3 | CPU_AVX512F | CPU_AVX512BW | CPU_AVX512CD | CPU_AVX512DQ |
             CPU_AVX512VL,
  };

  CPU_FEATURES cpu_feats = CPU_BASELINE;

  // When handling function arguments, we need to prevent argument registers
  // from being handed out as fixed registers
  //
  // Additionally, for now we prevent AX,DX,CX to be fixed to not run into
  // issues with instructions that need them as implicit arguments
  // also AX and DX can never be fixed if exception handling is used
  // since they are clobbered there
  u64 fixed_assignment_nonallocatable_mask =
      create_bitmask({AsmReg::AX, AsmReg::DX, AsmReg::CX});
  u32 func_start_off = 0u, func_prologue_alloc = 0u, func_epilogue_alloc = 0u;
  /// For vararg functions only: number of scalar and xmm registers used.
  // TODO: this information should be obtained from the CCAssigner.
  u32 scalar_arg_count = 0xFFFF'FFFF, vec_arg_count = 0xFFFF'FFFF;
  u32 reg_save_frame_off = 0;
  u32 var_arg_stack_off = 0;
  util::SmallVector<u32, 8> func_ret_offs = {};
  /// For functions without dynamic allocas, the largest size used for arguments
  /// passed on the stack to callees. This size is added to the stack pointer
  /// subtraction/addition in prologue/epilogue to avoid stack pointer
  /// adjustments at call sites.
  u32 max_callee_stack_arg_size;

  /// Whether flags must be preserved when materializing constants etc.
  bool preserve_flags;

  /// Symbol for __tls_get_addr.
  SymRef sym_tls_get_addr;

  class CallBuilder : public Base::template CallBuilderBase<CallBuilder> {
    u32 stack_adjust_off = 0;

    void set_stack_used() noexcept;

  public:
    CallBuilder(Derived &compiler, CCAssigner &assigner) noexcept
        : Base::template CallBuilderBase<CallBuilder>(compiler, assigner) {}

    void add_arg_byval(ValuePart &vp, CCAssignment &cca) noexcept;
    void add_arg_stack(ValuePart &vp, CCAssignment &cca) noexcept;
    void call_impl(std::variant<SymRef, ValuePart> &&target) noexcept;
    void reset_stack() noexcept;
  };

  // for now, always generate an object
  explicit CompilerX64(Adaptor *adaptor,
                       const CPU_FEATURES cpu_features = CPU_BASELINE)
      : Base{adaptor}, cpu_feats(cpu_features) {
    static_assert(std::is_base_of_v<CompilerX64, Derived>);
    static_assert(concepts::Compiler<Derived, PlatformConfig>);
  }

  template <typename... Args>
  auto asm_helper(unsigned (*enc_fn)(u8 *, int, Args...)) {
    struct Helper {
      CompilerX64 *compiler;
      decltype(enc_fn) fn;
      void encode(unsigned reserve, int flags, Args... args) {
        if (reserve) {
          compiler->text_writer.ensure_space(reserve);
        }
        unsigned n = fn(compiler->text_writer.cur_ptr(), flags, args...);
        assert(n != 0);
        compiler->text_writer.cur_ptr() += n;
      }
    };
    return Helper{this, enc_fn};
  }

  void start_func(u32 func_idx) noexcept;

  void gen_func_prolog_and_args(CCAssigner *) noexcept;

  void finish_func(u32 func_idx) noexcept;

  void reset() noexcept;

  // helpers

  void gen_func_epilog() noexcept;

  void set_preserve_flags(bool preserve) noexcept { preserve_flags = preserve; }
  bool may_clobber_flags() noexcept { return !preserve_flags; }

  void
      spill_reg(const AsmReg reg, const i32 frame_off, const u32 size) noexcept;

  void load_from_stack(AsmReg dst,
                       i32 frame_off,
                       u32 size,
                       bool sign_extend = false) noexcept;

  void load_address_of_stack_var(AsmReg dst, AssignmentPartRef ap) noexcept;

  void mov(AsmReg dst, AsmReg src, u32 size) noexcept;

  GenericValuePart val_spill_slot(AssignmentPartRef ap) noexcept {
    assert(ap.stack_valid() && !ap.variable_ref());
    return typename GenericValuePart::Expr(AsmReg::BP, ap.frame_off());
  }

  AsmReg gval_expr_as_reg(GenericValuePart &gv) noexcept;

  /// Dynamic alloca of a fixed-size region.
  void alloca_fixed(u64 size, u32 align, ValuePart &res) noexcept;

  /// Dynamic alloca of a dynamically-sized region (elem_size * count bytes).
  /// count must have a size of 64 bit.
  void alloca_dynamic(u64 elem_size,
                      ValuePart &&count,
                      u32 align,
                      ValuePart &res) noexcept;

  /// Materialize constant into a register.
  void materialize_constant(const u64 *data,
                            RegBank bank,
                            u32 size,
                            AsmReg dst) noexcept;

  AsmReg select_fixed_assignment_reg(AssignmentPartRef, IRValueRef) noexcept;

  /// Jump conditions.
  enum class Jump {
    jo = 0, ///< Jump if overflow (OF=1).
    jno,    ///< Jump if not overflow (OF=0).
    jb,     ///< Jump if below/if carry (CF=1).
    jae,    ///< Jump if above or equal/if not carry (CF=0).
    je,     ///< Jump if equal/if zero (ZF=1).
    jne,    ///< Jump if not equal/if not zero (ZF=0).
    jbe,    ///< Jump if below or equal (CF=1 or ZF=1).
    ja,     ///< Jump if above (CF=0 and ZF=0).
    js,     ///< Jump if sign (SF=1).
    jns,    ///< Jump if not sign (SF=0).
    jp,     ///< Jump if parity even (PF=1).
    jnp,    ///< Jump if parity odd (PF=0).
    jl,     ///< Jump if less (SF!=OF).
    jge,    ///< Jump if greater or equal (SF=OF).
    jle,    ///< Jump if less or equal (ZF=1 or SF!=OF).
    jg,     ///< Jump if greater (ZF=0 and SF=OF).
    jmp,    ///< Unconditional jump
  };

  Jump invert_jump(Jump jmp) noexcept;
  Jump swap_jump(Jump jmp) noexcept;

  FeCond jump_to_cond(Jump jmp) noexcept;

  void generate_branch_to_block(Jump jmp,
                                IRBlockRef target,
                                bool needs_split,
                                bool last_inst) noexcept;

  void generate_raw_jump(Jump jmp, Label target) noexcept;

  /// Set dst to 1 if cc is true, otherwise set it to zero. If zext is false,
  /// only the lowest 8 bit are set. Flags are not clobbered.
  void generate_raw_set(Jump cc, AsmReg dst, bool zext = true) noexcept;
  /// Set all bits of dst to 1 if cc is true, otherwise set it to zero
  void generate_raw_mask(Jump cc, AsmReg dst) noexcept;
  /// Move src into dst if cc is true, otherwise do nothing
  void generate_raw_cmov(Jump cc, AsmReg dst, AsmReg src, bool is_64) noexcept;

  /// Integer extension. Might need a temporary register, src is not modified,
  /// might clobber flags.
  void generate_raw_intext(
      AsmReg dst, AsmReg src, bool sign, u32 from, u32 to) noexcept;

  /// Bitfield insert. Needs a temporary register, src is not modified.
  void generate_raw_bfi(AsmReg dst, AsmReg src, u32 lsb, u32 width) noexcept;
  /// Bitfield insert in zero. src is not modified, but src and dst must be
  /// different.
  void generate_raw_bfiz(AsmReg dst, AsmReg src, u32 lsb, u32 width) noexcept;

  /// Generate a function call
  ///
  /// This will get the arguments into the correct registers according to the
  /// calling convention, clear non-callee-saved registers from the register
  /// file (make sure you do not have any fixed assignments left over) and
  /// fill the result registers (the u8 in the ScratchReg pair indicates the
  /// register bank)
  ///
  /// Targets can be a symbol (call to PLT with relocation), or an indirect
  /// call to a ValuePart. Result is an optional reference.
  void generate_call(std::variant<SymRef, ValuePart> &&target,
                     std::span<CallArg> arguments,
                     typename Base::ValueRef *result,
                     bool variable_args = false);

private:
  /// @internal Emit compare of cmp_reg with case_value.
  void switch_emit_cmp(AsmReg cmp_reg,
                       AsmReg tmp_reg,
                       u64 case_value,
                       bool width_is_32) noexcept;

public:
  /// @internal Jump if cmp_reg equals case_value.
  void switch_emit_cmpeq(Label case_label,
                         AsmReg cmp_reg,
                         AsmReg tmp_reg,
                         u64 case_value,
                         bool width_is_32) noexcept;
  /// @internal Emit bounds check and jump table.
  bool switch_emit_jump_table(Label default_label,
                              std::span<const Label> labels,
                              AsmReg cmp_reg,
                              AsmReg tmp_reg,
                              u64 low_bound,
                              u64 high_bound,
                              bool width_is_32) noexcept;
  /// @internal Jump if cmp_reg is greater than case_value.
  void switch_emit_binary_step(Label case_label,
                               Label gt_label,
                               AsmReg cmp_reg,
                               AsmReg tmp_reg,
                               u64 case_value,
                               bool width_is_32) noexcept;

  /// Generate code sequence to load address of sym into a register. This will
  /// generate a function call for dynamic TLS access models.
  ScratchReg tls_get_addr(SymRef sym, TLSModel model) noexcept;

  bool has_cpu_feats(CPU_FEATURES feats) const noexcept {
    return ((cpu_feats & feats) == feats);
  }
};

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> class BaseTy,
          typename Config>
void CompilerX64<Adaptor, Derived, BaseTy, Config>::start_func(
    const u32 /*func_idx*/) noexcept {
  this->text_writer.align(16);
  this->assembler.except_begin_func();
  this->preserve_flags = false;
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
void CompilerX64<Adaptor, Derived, BaseTy, Config>::gen_func_prolog_and_args(
    CCAssigner *cc_assigner) noexcept {
  // prologue:
  // push rbp
  // mov rbp, rsp
  // optionally create vararg save-area
  // reserve space for callee-saved regs
  //   = 1 byte for each of the lower 8 regs and 2
  //   bytes for the higher 8 regs
  // sub rsp, #<frame_size>+<largest_call_frame_usage>

  // TODO(ts): technically we only need rbp if there
  // is a dynamic alloca but then we need to make the
  // frame indexing dynamic in CompilerBase and the
  // unwind info needs to take the dynamic sub rsp for
  // calls into account

  func_ret_offs.clear();
  func_start_off = this->text_writer.offset();
  scalar_arg_count = vec_arg_count = 0xFFFF'FFFF;

  const CCInfo &cc_info = cc_assigner->get_ccinfo();

  auto csr = cc_info.callee_saved_regs;
  assert(!(csr & ~this->register_file.bank_regs(Config::GP_BANK)) &&
         "non-gp callee-saved registers not implemented");

  u32 csr_logp = std::popcount((csr >> AsmReg::AX) & 0xff);
  u32 csr_higp = std::popcount((csr >> AsmReg::R8) & 0xff);
  // R8 and higher need a REX prefix
  u32 reg_save_size = 1 * csr_logp + 2 * csr_higp;
  this->stack.frame_size = 8 * (csr_logp + csr_higp);
  max_callee_stack_arg_size = 0;

  // 11 bytes for push rbp/mov rbp, rsp/sub rsp
  func_prologue_alloc = reg_save_size + 11;
  this->text_writer.ensure_space(func_prologue_alloc);
  this->text_writer.cur_ptr() += func_prologue_alloc;
  // pop has the same size as push; add/sub reg,imm32 and lea rsp, [rbp-imm32]
  // have 7 bytes; pop rbp has 1 byte; ret has 1 byte.
  func_epilogue_alloc = reg_save_size + 9;

  // TODO(ts): support larger stack alignments?

  if (this->adaptor->cur_is_vararg()) {
    this->stack.frame_used = true;
    this->stack.frame_size += 6 * 8 + 8 * 16;
    reg_save_frame_off = this->stack.frame_size;
    auto mem = FE_MEM(FE_BP, 0, FE_NOREG, -(i32)reg_save_frame_off);
    ASM(MOV64mr, mem, FE_DI);
    mem.off += 8;
    ASM(MOV64mr, mem, FE_SI);
    mem.off += 8;
    ASM(MOV64mr, mem, FE_DX);
    mem.off += 8;
    ASM(MOV64mr, mem, FE_CX);
    mem.off += 8;
    ASM(MOV64mr, mem, FE_R8);
    mem.off += 8;
    ASM(MOV64mr, mem, FE_R9);
    auto skip_fp = this->text_writer.label_create();
    ASM(TEST8rr, FE_AX, FE_AX);
    generate_raw_jump(Jump::je, skip_fp);
    mem.off += 8;
    ASM(SSE_MOVDQUmr, mem, FE_XMM0);
    mem.off += 16;
    ASM(SSE_MOVDQUmr, mem, FE_XMM1);
    mem.off += 16;
    ASM(SSE_MOVDQUmr, mem, FE_XMM2);
    mem.off += 16;
    ASM(SSE_MOVDQUmr, mem, FE_XMM3);
    mem.off += 16;
    ASM(SSE_MOVDQUmr, mem, FE_XMM4);
    mem.off += 16;
    ASM(SSE_MOVDQUmr, mem, FE_XMM5);
    mem.off += 16;
    ASM(SSE_MOVDQUmr, mem, FE_XMM6);
    mem.off += 16;
    ASM(SSE_MOVDQUmr, mem, FE_XMM7);
    this->label_place(skip_fp);
  }

  // Temporarily prevent argument registers from being assigned.
  assert((cc_info.allocatable_regs & cc_info.arg_regs) == cc_info.arg_regs &&
         "argument registers must also be allocatable");
  this->register_file.allocatable &= ~cc_info.arg_regs;

  u32 arg_idx = 0;
  for (const IRValueRef arg : this->adaptor->cur_args()) {
    derived()->handle_func_arg(
        arg_idx,
        arg,
        [&](ValuePart &&vp, CCAssignment cca) -> std::optional<i32> {
          if (!cca.byval) {
            cca.bank = vp.bank();
            cca.size = vp.part_size();
          }

          cc_assigner->assign_arg(cca);

          if (cca.reg.valid()) [[likely]] {
            vp.set_value_reg(this, cca.reg);
            // Mark register as allocatable as soon as it is assigned. If the
            // argument is unused, the register will be freed immediately and
            // can be used for later stack arguments.
            this->register_file.allocatable |= u64{1} << cca.reg.id();
            return {};
          }

          if (vp.is_owned()) {
            // no need to handle unused arguments
            return {};
          }

          this->stack.frame_used = true;
          if (cca.byval) {
            // Return byval frame_off.
            return 0x10 + cca.stack_off;
          } else {
            //  TODO(ts): maybe allow negative frame offsets for value
            //  assignments so we can simply reference this?
            //  but this probably doesn't work with multi-part values
            //  since the offsets are different
            AsmReg dst = vp.alloc_reg(this);
            this->load_from_stack(dst, 0x10 + cca.stack_off, cca.size);
          }
          return {};
        });

    arg_idx += 1;
  }

  if (this->adaptor->cur_is_vararg()) [[unlikely]] {
    // TODO: get this from CCAssigner?
    auto arg_regs = this->register_file.allocatable & cc_info.arg_regs;
    u64 gp_regs = arg_regs & this->register_file.bank_regs(Config::GP_BANK);
    u64 xmm_regs = arg_regs & this->register_file.bank_regs(Config::FP_BANK);
    this->scalar_arg_count = std::popcount(gp_regs);
    this->vec_arg_count = std::popcount(xmm_regs);
    this->var_arg_stack_off = 0x10 + cc_assigner->get_stack_size();
  }

  this->register_file.allocatable |= cc_info.arg_regs;
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
void CompilerX64<Adaptor, Derived, BaseTy, Config>::finish_func(
    u32 func_idx) noexcept {
  // NB: code alignment factor 1, data alignment factor -8.
  auto fde_off = this->assembler.eh_begin_fde(this->get_personality_sym());
  // push rbp
  this->assembler.eh_write_inst(dwarf::DW_CFA_advance_loc, 1);
  this->assembler.eh_write_inst(dwarf::DW_CFA_def_cfa_offset, 16);
  this->assembler.eh_write_inst(
      dwarf::DW_CFA_offset, dwarf::x64::DW_reg_rbp, 2);
  // mov rbp, rsp
  this->assembler.eh_write_inst(dwarf::DW_CFA_advance_loc, 3);
  this->assembler.eh_write_inst(dwarf::DW_CFA_def_cfa_register,
                                dwarf::x64::DW_reg_rbp);

  // Patched below
  auto fde_prologue_adv_off = this->assembler.eh_writer.size();
  this->assembler.eh_write_inst(dwarf::DW_CFA_advance_loc, 0);

  auto *write_ptr = this->text_writer.begin_ptr() + func_start_off;
  write_ptr += fe64_PUSHr(write_ptr, 0, FE_BP);
  write_ptr += fe64_MOV64rr(write_ptr, 0, FE_BP, FE_SP);
  const CCInfo &ccinfo = derived()->cur_cc_assigner()->get_ccinfo();
  auto csr = ccinfo.callee_saved_regs;
  u64 saved_regs = this->register_file.clobbered & csr;
  u32 num_saved_regs = 0u;
  for (auto reg : util::BitSetIterator{saved_regs}) {
    assert(reg <= AsmReg::R15);
    write_ptr +=
        fe64_PUSHr(write_ptr, 0, AsmReg{static_cast<AsmReg::REG>(reg)});
    ++num_saved_regs;

    // DWARF register ordering is subtly different from the encoding:
    // x86 is:   ax, cx, dx, bx, sp, bp, si, di, r8, ...
    // DWARF is: ax, dx, cx, bx, si, di, bp, sp, r8, ...
    static const u8 gpreg_to_dwarf[] = {
        dwarf::x64::DW_reg_rax,
        dwarf::x64::DW_reg_rcx,
        dwarf::x64::DW_reg_rdx,
        dwarf::x64::DW_reg_rbx,
        dwarf::x64::DW_reg_rsp,
        dwarf::x64::DW_reg_rbp,
        dwarf::x64::DW_reg_rsi,
        dwarf::x64::DW_reg_rdi,
        dwarf::x64::DW_reg_r8,
        dwarf::x64::DW_reg_r9,
        dwarf::x64::DW_reg_r10,
        dwarf::x64::DW_reg_r11,
        dwarf::x64::DW_reg_r12,
        dwarf::x64::DW_reg_r13,
        dwarf::x64::DW_reg_r14,
        dwarf::x64::DW_reg_r15,
    };
    u8 dwarf_reg = gpreg_to_dwarf[reg];
    auto cfa_off = num_saved_regs + 2;
    this->assembler.eh_write_inst(dwarf::DW_CFA_offset, dwarf_reg, cfa_off);
  }

  assert((!this->stack.has_dynamic_alloca || max_callee_stack_arg_size == 0) &&
         "stack with dynamic alloca must adjust stack pointer at call sites");
  // The frame_size contains the reserved frame size so we need to subtract
  // the stack space we used for the saved registers
  u32 final_frame_size =
      util::align_up(this->stack.frame_size + max_callee_stack_arg_size, 16);
  u32 rsp_adjustment = final_frame_size - num_saved_regs * 8;
  bool needs_rsp_adjustment = this->stack.generated_call ||
                              this->stack.has_dynamic_alloca ||
                              rsp_adjustment > ccinfo.red_zone_size;

  if (needs_rsp_adjustment) {
    write_ptr += fe64_SUB64ri(write_ptr, 0, FE_SP, rsp_adjustment);
  }

  u32 prologue_size =
      write_ptr - (this->text_writer.begin_ptr() + func_start_off);
  assert(prologue_size <= func_prologue_alloc);
  assert(prologue_size < 0x44 && "cannot encode too large prologue in DWARF");
  this->assembler.eh_writer.data()[fde_prologue_adv_off] =
      dwarf::DW_CFA_advance_loc | (prologue_size - 4);

  // nop out the rest
  if (u32 nop_len = func_prologue_alloc - prologue_size) {
    fe64_NOP(write_ptr, nop_len);
  }

  auto func_sym = this->func_syms[func_idx];
  auto func_sec = this->text_writer.get_sec_ref();
  if (func_ret_offs.empty()) {
    // TODO(ts): honor cur_needs_unwind_info
    auto func_size = this->text_writer.offset() - func_start_off;
    this->assembler.sym_def(func_sym, func_sec, func_start_off, func_size);
    this->assembler.eh_end_fde(fde_off, func_sym);
    this->assembler.except_encode_func(func_sym,
                                       this->text_writer.label_offsets.data());
    return;
  }

  auto *text_data = this->text_writer.begin_ptr();
  u32 first_ret_off = func_ret_offs[0];
  u32 ret_size = 0;
  u32 epilogue_size = func_epilogue_alloc;
  u32 func_end_ret_off = this->text_writer.offset() - epilogue_size;
  {
    write_ptr = text_data + first_ret_off;
    const auto ret_start = write_ptr;
    if (this->stack.has_dynamic_alloca) {
      if (num_saved_regs == 0) {
        write_ptr += fe64_MOV64rr(write_ptr, 0, FE_SP, FE_BP);
      } else {
        write_ptr +=
            fe64_LEA64rm(write_ptr,
                         0,
                         FE_SP,
                         FE_MEM(FE_BP, 0, FE_NOREG, -(i32)num_saved_regs * 8));
      }
    } else if (needs_rsp_adjustment) {
      write_ptr += fe64_ADD64ri(write_ptr, 0, FE_SP, rsp_adjustment);
    }
    for (auto reg : util::BitSetIterator<true>{saved_regs}) {
      assert(reg <= AsmReg::R15);
      write_ptr +=
          fe64_POPr(write_ptr, 0, AsmReg{static_cast<AsmReg::REG>(reg)});
    }
    write_ptr += fe64_POPr(write_ptr, 0, FE_BP);
    write_ptr += fe64_RET(write_ptr, 0);
    ret_size = write_ptr - ret_start;
    assert(ret_size <= epilogue_size && "function epilogue too long");

    // write NOP for better disassembly
    if (epilogue_size > ret_size) {
      fe64_NOP(write_ptr, epilogue_size - ret_size);
      if (first_ret_off == func_end_ret_off) {
        this->text_writer.cur_ptr() -= epilogue_size - ret_size;
      }
    }
  }

  for (u32 i = 1; i < func_ret_offs.size(); ++i) {
    std::memcpy(
        text_data + func_ret_offs[i], text_data + first_ret_off, epilogue_size);
    if (func_ret_offs[i] == func_end_ret_off) {
      this->text_writer.cur_ptr() -= epilogue_size - ret_size;
    }
  }

  // Do sym_def at the very end; we shorten the function here again, so only at
  // this point we know the actual size of the function.
  // TODO(ts): honor cur_needs_unwind_info
  auto func_size = this->text_writer.offset() - func_start_off;
  this->assembler.sym_def(func_sym, func_sec, func_start_off, func_size);
  this->assembler.eh_end_fde(fde_off, func_sym);
  this->assembler.except_encode_func(func_sym,
                                     this->text_writer.label_offsets.data());
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
void CompilerX64<Adaptor, Derived, BaseTy, Config>::reset() noexcept {
  func_ret_offs.clear();
  sym_tls_get_addr = {};
  Base::reset();
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
void CompilerX64<Adaptor, Derived, BaseTy, Config>::gen_func_epilog() noexcept {
  // Patched at the end, just reserve the space here.
  func_ret_offs.push_back(this->text_writer.offset());
  this->text_writer.ensure_space(func_epilogue_alloc);
  this->text_writer.cur_ptr() += func_epilogue_alloc;
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
void CompilerX64<Adaptor, Derived, BaseTy, Config>::spill_reg(
    const AsmReg reg, const i32 frame_off, const u32 size) noexcept {
  assert(this->stack.frame_used);
  this->text_writer.ensure_space(16);
  assert(frame_off < 0);
  const auto mem = FE_MEM(FE_BP, 0, FE_NOREG, frame_off);
  if (reg.id() <= AsmReg::R15) {
    switch (size) {
    case 1: ASMNC(MOV8mr, mem, reg); break;
    case 2: ASMNC(MOV16mr, mem, reg); break;
    case 4: ASMNC(MOV32mr, mem, reg); break;
    case 8: ASMNC(MOV64mr, mem, reg); break;
    default: TPDE_UNREACHABLE("invalid spill size");
    }
    return;
  }

  switch (size) {
  case 4: ASMNC(SSE_MOVD_X2Gmr, mem, reg); break;
  case 8: ASMNC(SSE_MOVQ_X2Gmr, mem, reg); break;
  case 16: ASMNC(SSE_MOVAPDmr, mem, reg); break;
  default: TPDE_UNREACHABLE("invalid spill size");
  }
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
void CompilerX64<Adaptor, Derived, BaseTy, Config>::load_from_stack(
    const AsmReg dst,
    const i32 frame_off,
    const u32 size,
    const bool sign_extend) noexcept {
  assert(this->stack.frame_used);
  this->text_writer.ensure_space(16);
  const auto mem = FE_MEM(FE_BP, 0, FE_NOREG, frame_off);

  if (dst.id() <= AsmReg::R15) {
    if (!sign_extend) {
      switch (size) {
      case 1: ASMNC(MOVZXr32m8, dst, mem); break;
      case 2: ASMNC(MOVZXr32m16, dst, mem); break;
      case 4: ASMNC(MOV32rm, dst, mem); break;
      case 8: ASMNC(MOV64rm, dst, mem); break;
      default: TPDE_UNREACHABLE("invalid spill size");
      }
    } else {
      switch (size) {
      case 1: ASMNC(MOVSXr64m8, dst, mem); break;
      case 2: ASMNC(MOVSXr64m16, dst, mem); break;
      case 4: ASMNC(MOVSXr64m32, dst, mem); break;
      case 8: ASMNC(MOV64rm, dst, mem); break;
      default: TPDE_UNREACHABLE("invalid spill size");
      }
    }
    return;
  }

  assert(!sign_extend);

  switch (size) {
  case 4: ASMNC(SSE_MOVD_G2Xrm, dst, mem); break;
  case 8: ASMNC(SSE_MOVQ_G2Xrm, dst, mem); break;
  case 16: ASMNC(SSE_MOVAPDrm, dst, mem); break;
  default: TPDE_UNREACHABLE("invalid spill size");
  }
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
void CompilerX64<Adaptor, Derived, BaseTy, Config>::load_address_of_stack_var(
    const AsmReg dst, const AssignmentPartRef ap) noexcept {
  assert(this->stack.frame_used);
  ASM(LEA64rm, dst, FE_MEM(FE_BP, 0, FE_NOREG, ap.variable_stack_off()));
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
void CompilerX64<Adaptor, Derived, BaseTy, Config>::mov(
    const AsmReg dst, const AsmReg src, const u32 size) noexcept {
  this->text_writer.ensure_space(16);
  assert(dst.valid());
  assert(src.valid());
  if (dst.id() <= AsmReg::R15 && src.id() <= AsmReg::R15) {
    if (size > 4) {
      ASMNC(MOV64rr, dst, src);
    } else {
      ASMNC(MOV32rr, dst, src);
    }
  } else if (dst.id() >= AsmReg::XMM0 && src.id() >= AsmReg::XMM0) {
    if (size <= 16) {
      if (dst.id() > AsmReg::XMM15 || src.id() > AsmReg::XMM15) {
        assert(has_cpu_feats(CPU_AVX512F));
        ASMNC(VMOVAPD128rr, dst, src);
      } else {
        ASMNC(SSE_MOVAPDrr, dst, src);
      }
    } else if (size <= 32) {
      assert(has_cpu_feats(CPU_AVX));
      assert((dst.id() <= AsmReg::XMM15 && src.id() <= AsmReg::XMM15) ||
             has_cpu_feats(CPU_AVX512F));
      ASMNC(VMOVAPD256rr, dst, src);
    } else {
      assert(size <= 64);
      assert(has_cpu_feats(CPU_AVX512F));
      ASMNC(VMOVAPD512rr, dst, src);
    }
  } else if (dst.id() <= AsmReg::R15) {
    // gp<-xmm
    assert(src.id() >= AsmReg::XMM0);
    assert(size <= 8);
    if (src.id() > AsmReg::XMM15) {
      assert(has_cpu_feats(CPU_AVX512F));
      if (size <= 4) {
        ASMNC(VMOVD_X2Grr, dst, src);
      } else {
        ASMNC(VMOVQ_X2Grr, dst, src);
      }
    } else {
      if (size <= 4) {
        ASMNC(SSE_MOVD_X2Grr, dst, src);
      } else {
        ASMNC(SSE_MOVQ_X2Grr, dst, src);
      }
    }
  } else {
    // xmm<-gp
    assert(src.id() <= AsmReg::R15);
    assert(dst.id() >= AsmReg::XMM0);
    assert(size <= 8);
    if (dst.id() > AsmReg::XMM15) {
      assert(has_cpu_feats(CPU_AVX512F));
      if (size <= 4) {
        ASMNC(VMOVD_G2Xrr, dst, src);
      } else {
        ASMNC(VMOVQ_G2Xrr, dst, src);
      }
    } else {
      if (size <= 4) {
        ASMNC(SSE_MOVD_G2Xrr, dst, src);
      } else {
        ASMNC(SSE_MOVQ_G2Xrr, dst, src);
      }
    }
  }
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
AsmReg CompilerX64<Adaptor, Derived, BaseTy, Config>::gval_expr_as_reg(
    GenericValuePart &gv) noexcept {
  auto &expr = std::get<typename GenericValuePart::Expr>(gv.state);

  ScratchReg scratch{derived()};
  bool disp32 = i32(expr.disp) == expr.disp;
  AsmReg base = expr.has_base() ? expr.base_reg() : AsmReg::make_invalid();
  AsmReg idx = expr.has_index() ? expr.index_reg() : AsmReg::make_invalid();
  if (std::holds_alternative<ScratchReg>(expr.base)) {
    scratch = std::move(std::get<ScratchReg>(expr.base));
  } else if (std::holds_alternative<ScratchReg>(expr.index)) {
    scratch = std::move(std::get<ScratchReg>(expr.index));
  } else {
    (void)scratch.alloc_gp();
  }
  auto dst = scratch.cur_reg();
  if (idx.valid()) {
    if ((expr.scale & (expr.scale - 1)) == 0 && expr.scale < 16) {
      u8 sc = expr.scale;
      if (base.valid() && disp32) {
        ASM(LEA64rm, dst, FE_MEM(base, sc, idx, i32(expr.disp)));
        expr.disp = 0;
      } else if (base.valid()) {
        ASM(LEA64rm, dst, FE_MEM(base, sc, idx, 0));
      } else if (disp32) {
        ASM(LEA64rm, dst, FE_MEM(FE_NOREG, sc, idx, i32(expr.disp)));
      } else {
        ASM(LEA64rm, dst, FE_MEM(FE_NOREG, sc, idx, 0));
      }
    } else {
      assert(may_clobber_flags());
      u64 scale = expr.scale;
      if (base == idx) {
        base = AsmReg::make_invalid();
        scale += 1;
      }

      ScratchReg idx_scratch{derived()};
      // We need a register to compute the scaled index.
      AsmReg idx_tmp = dst;
      if (dst == base && std::holds_alternative<ScratchReg>(expr.index)) {
        // We can't use dst, it'd clobber base, so use the other
        // register we currently own.
        idx_tmp = std::get<ScratchReg>(expr.index).cur_reg();
      } else if (dst == base) {
        idx_tmp = idx_scratch.alloc_gp();
      }

      if ((scale & (scale - 1)) == 0) {
        if (idx_tmp != idx) {
          ASM(MOV64rr, idx_tmp, idx);
        }
        ASM(SHL64ri, idx_tmp, util::cnt_tz(scale));
      } else {
        if (i32(scale) == i64(scale)) {
          ASM(IMUL64rri, idx_tmp, idx, scale);
        } else {
          ScratchReg scratch2{derived()};
          auto tmp2 = scratch2.alloc_gp();
          ASM(MOV64ri, tmp2, scale);
          if (idx_tmp != idx) {
            ASM(MOV64rr, idx_tmp, idx);
          }
          ASM(IMUL64rr, idx_tmp, tmp2);
        }
      }
      if (base.valid()) {
        if (disp32 || (idx_tmp != dst && base != dst)) {
          ASM(LEA64rm, dst, FE_MEM(base, 1, idx_tmp, i32(expr.disp)));
          expr.disp = 0;
        } else if (dst == base) {
          ASM(ADD64rr, dst, idx_tmp);
        } else {
          ASM(ADD64rr, dst, base);
        }
      }
    }
  } else if (base.valid()) {
    if (expr.disp && disp32) {
      ASM(LEA64rm, dst, FE_MEM(base, 0, FE_NOREG, i32(expr.disp)));
      expr.disp = 0;
    } else if (dst != base) {
      ASM(MOV64rr, dst, base);
    }
  }
  if (expr.disp) {
    ScratchReg scratch2{derived()};
    auto tmp2 = scratch2.alloc_gp();
    ASM(MOV64ri, tmp2, expr.disp);
    if (may_clobber_flags()) {
      ASM(ADD64rr, dst, tmp2);
    } else {
      ASM(LEA64rm, dst, FE_MEM(dst, 1, tmp2, 0));
    }
  }
  gv.state = std::move(scratch);
  return dst;
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
void CompilerX64<Adaptor, Derived, BaseTy, Config>::alloca_fixed(
    u64 size, u32 align, ValuePart &res) noexcept {
  assert(this->stack.has_dynamic_alloca &&
         "function marked as not having dynamic allocas can't have alloca");
  assert(align != 0 && (align & (align - 1)) == 0 && "invalid alignment");
  assert(may_clobber_flags());
  size = tpde::util::align_up(size, 16);
  if (size > 0) {
    assert(size < 0x8000'0000);
    ASM(SUB64ri, FE_SP, size);
  }
  if (align > 16) {
    assert(align < u32{1} << 31 && "alignment >= 2**31 not implemented");
    ASM(AND64ri, FE_SP, ~(align - 1));
  }
  ASM(MOV64rr, res.alloc_reg(this), FE_SP);
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
void CompilerX64<Adaptor, Derived, BaseTy, Config>::alloca_dynamic(
    u64 elem_size, ValuePart &&count, u32 align, ValuePart &res) noexcept {
  assert(this->stack.has_dynamic_alloca &&
         "function marked as not having dynamic allocas can't have alloca");
  assert(align != 0 && (align & (align - 1)) == 0 && "invalid alignment");
  assert(may_clobber_flags());
  AsmReg size_reg = count.has_reg() ? count.cur_reg() : count.load_to_reg(this);
  AsmReg res_reg = res.alloc_try_reuse(this, count);

  if (elem_size == 0) {
    ASM(XOR32rr, res_reg, res_reg);
  } else if ((elem_size & (elem_size - 1)) == 0) {
    // elem_size is power of two
    const auto shift = util::cnt_tz(elem_size);
    if (shift > 0 && shift < 4) {
      ASM(LEA64rm, res_reg, FE_MEM(FE_NOREG, u8(1 << shift), size_reg, 0));
    } else {
      if (size_reg != res_reg) {
        ASM(MOV64rr, res_reg, size_reg);
      }
      if (elem_size != 1) {
        ASM(SHL64ri, res_reg, shift);
      }
    }
  } else {
    if (elem_size <= 0x7FFF'FFFF) [[likely]] {
      ASM(IMUL64rri, res_reg, size_reg, elem_size);
    } else {
      ScratchReg scratch{this};
      auto tmp = scratch.alloc_gp();
      ASM(MOV64ri, tmp, elem_size);
      if (size_reg != res_reg) {
        ASM(MOV64rr, res_reg, size_reg);
      }
      ASM(IMUL64rr, res_reg, tmp);
    }
  }

  ASM(SUB64rr, FE_SP, res_reg);

  align = align > 16 ? align : 16;
  if (elem_size & (align - 1)) {
    assert(align < u32{1} << 31 && "alignment >= 2**31 not implemented");
    ASM(AND64ri, FE_SP, ~(align - 1));
  }

  ASM(MOV64rr, res_reg, FE_SP);
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
void CompilerX64<Adaptor, Derived, BaseTy, Config>::materialize_constant(
    const u64 *data, const RegBank bank, const u32 size, AsmReg dst) noexcept {
  const auto const_u64 = data[0];
  if (bank == Config::GP_BANK) {
    assert(size <= 8);
    if (const_u64 == 0) {
      if (may_clobber_flags()) {
        ASM(XOR32rr, dst, dst);
      } else {
        ASM(MOV32ri, dst, 0);
      }
      return;
    }

    if (size <= 4 || u32(const_u64) == const_u64) {
      ASM(MOV32ri, dst, const_u64);
    } else {
      ASM(MOV64ri, dst, const_u64);
    }
    return;
  }

  assert(bank == Config::FP_BANK);
  const auto high_u64 = size <= 8 ? 0 : data[1];
  if (const_u64 == 0 && (size <= 8 || (high_u64 == 0 && size <= 16))) {
    if (has_cpu_feats(CPU_AVX)) {
      ASM(VPXOR128rrr, dst, dst, dst);
    } else {
      ASM(SSE_PXORrr, dst, dst);
    }
    return;
  }
  const u64 ones = -u64{1};
  if (const_u64 == ones && (size <= 8 || (high_u64 == ones && size <= 16))) {
    if (has_cpu_feats(CPU_AVX)) {
      ASM(VPCMPEQB128rrr, dst, dst, dst);
    } else {
      ASM(SSE_PCMPEQBrr, dst, dst);
    }
    return;
  }

  if (size <= 8) {
    // We must not evict registers here (might be used within branching code),
    // so only use free registers and load from memory otherwise.
    AsmReg tmp =
        this->register_file.find_first_free_excluding(Config::GP_BANK, 0);
    if (tmp.valid()) {
      this->register_file.mark_clobbered(tmp);
      materialize_constant(data, Config::GP_BANK, size, tmp);
      if (size <= 4) {
        if (has_cpu_feats(CPU_AVX)) {
          ASM(VMOVD_G2Xrr, dst, tmp);
        } else {
          ASM(SSE_MOVD_G2Xrr, dst, tmp);
        }
      } else {
        if (has_cpu_feats(CPU_AVX)) {
          ASM(VMOVQ_G2Xrr, dst, tmp);
        } else {
          ASM(SSE_MOVQ_G2Xrr, dst, tmp);
        }
      }
      return;
    }
  }

  // TODO: round to next power of two but at least 4 byte
  // We store constants in 8-byte units.
  auto alloc_size = util::align_up(size, 8);
  std::span<const u8> raw_data{reinterpret_cast<const u8 *>(data), alloc_size};
  // TODO: deduplicate/pool constants?
  auto rodata = this->assembler.get_data_section(true, false);
  auto sym = this->assembler.sym_def_data(
      rodata, "", raw_data, alloc_size, Assembler::SymBinding::LOCAL);
  if (size <= 4) {
    if (has_cpu_feats(CPU_AVX)) {
      ASM(VMOVSSrm, dst, FE_MEM(FE_IP, 0, FE_NOREG, -1));
    } else {
      ASM(SSE_MOVSSrm, dst, FE_MEM(FE_IP, 0, FE_NOREG, -1));
    }
  } else if (size <= 8) {
    if (has_cpu_feats(CPU_AVX)) {
      ASM(VMOVSDrm, dst, FE_MEM(FE_IP, 0, FE_NOREG, -1));
    } else {
      ASM(SSE_MOVSDrm, dst, FE_MEM(FE_IP, 0, FE_NOREG, -1));
    }
  } else if (size <= 16) {
    if (has_cpu_feats(CPU_AVX)) {
      ASM(VMOVAPS128rm, dst, FE_MEM(FE_IP, 0, FE_NOREG, -1));
    } else {
      ASM(SSE_MOVAPSrm, dst, FE_MEM(FE_IP, 0, FE_NOREG, -1));
    }
  } else {
    // TODO: implement for AVX/AVX-512.
    TPDE_FATAL("unable to materialize constant");
  }

  this->reloc_text(sym, R_X86_64_PC32, this->text_writer.offset() - 4, -4);
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
AsmReg
    CompilerX64<Adaptor, Derived, BaseTy, Config>::select_fixed_assignment_reg(
        AssignmentPartRef ap, IRValueRef) noexcept {
  RegBank bank = ap.bank();
  assert(bank.id() <= Config::NUM_BANKS);
  auto reg_mask = this->register_file.bank_regs(bank);
  reg_mask &= ~fixed_assignment_nonallocatable_mask;

  const auto find_possible_regs = [this,
                                   reg_mask](const u64 preferred_regs) -> u64 {
    // try to first get an unused reg, otherwise an unfixed reg
    u64 free_regs = this->register_file.allocatable & ~this->register_file.used;
    return free_regs & preferred_regs & reg_mask;
  };

  u64 possible_regs;
  auto csr = derived()->cur_cc_assigner()->get_ccinfo().callee_saved_regs;
  if (!this->stack.is_leaf_function) {
    // we can only allocated fixed assignments from the callee-saved regs
    possible_regs = find_possible_regs(csr);
  } else {
    // try allocating any non-callee saved register first, except the result
    // registers
    possible_regs = find_possible_regs(~csr);
    if (possible_regs == 0) {
      // otherwise fallback to callee-saved regs
      possible_regs = find_possible_regs(csr);
    }
  }

  if (possible_regs == 0) {
    return AsmReg::make_invalid();
  }

  // try to first get an unused reg, otherwise an unfixed reg
  if ((possible_regs & ~this->register_file.used) != 0) {
    return AsmReg{util::cnt_tz(possible_regs & ~this->register_file.used)};
  }

  for (const auto reg_id : util::BitSetIterator<>{possible_regs}) {
    const auto reg = AsmReg{reg_id};

    if (this->register_file.is_fixed(reg)) {
      continue;
    }

    const auto local_idx = this->register_file.reg_local_idx(reg);
    const auto part = this->register_file.reg_part(reg);

    if (local_idx == Base::INVALID_VAL_LOCAL_IDX) {
      continue;
    }
    auto *assignment = this->val_assignment(local_idx);
    auto ap = AssignmentPartRef{assignment, part};
    if (ap.modified()) {
      continue;
    }

    return reg;
  }

  return AsmReg::make_invalid();
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
typename CompilerX64<Adaptor, Derived, BaseTy, Config>::Jump
    CompilerX64<Adaptor, Derived, BaseTy, Config>::invert_jump(
        Jump jmp) noexcept {
  switch (jmp) {
  case Jump::ja: return Jump::jbe;
  case Jump::jae: return Jump::jb;
  case Jump::jb: return Jump::jae;
  case Jump::jbe: return Jump::ja;
  case Jump::je: return Jump::jne;
  case Jump::jg: return Jump::jle;
  case Jump::jge: return Jump::jl;
  case Jump::jl: return Jump::jge;
  case Jump::jle: return Jump::jg;
  case Jump::jne: return Jump::je;
  case Jump::jno: return Jump::jo;
  case Jump::jo: return Jump::jno;
  case Jump::js: return Jump::jns;
  case Jump::jns: return Jump::js;
  case Jump::jp: return Jump::jnp;
  case Jump::jnp: return Jump::jp;
  default: TPDE_UNREACHABLE("invalid jump kind for invert_jump");
  }
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> class BaseTy,
          typename Config>
typename CompilerX64<Adaptor, Derived, BaseTy, Config>::Jump
    CompilerX64<Adaptor, Derived, BaseTy, Config>::swap_jump(
        Jump jmp) noexcept {
  switch (jmp) {
  case Jump::ja: return Jump::jb;
  case Jump::jae: return Jump::jbe;
  case Jump::jb: return Jump::ja;
  case Jump::jbe: return Jump::jae;
  case Jump::je: return Jump::je;
  case Jump::jne: return Jump::jne;
  case Jump::jg: return Jump::jl;
  case Jump::jge: return Jump::jle;
  case Jump::jl: return Jump::jg;
  case Jump::jle: return Jump::jge;
  default: TPDE_UNREACHABLE("invalid jump kind for swap_jump");
  }
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> class BaseTy,
          typename Config>
FeCond CompilerX64<Adaptor, Derived, BaseTy, Config>::jump_to_cond(
    Jump jmp) noexcept {
  // LLVM won't transform the switch into a shift.
  FeCond res = FeCond(u32(jmp) << 16);
  switch (jmp) {
  case Jump::ja: assert(res == FE_CC_A && "FeCond value mismatch?"); break;
  case Jump::jae: assert(res == FE_CC_AE && "FeCond value mismatch?"); break;
  case Jump::jb: assert(res == FE_CC_B && "FeCond value mismatch?"); break;
  case Jump::jbe: assert(res == FE_CC_BE && "FeCond value mismatch?"); break;
  case Jump::je: assert(res == FE_CC_E && "FeCond value mismatch?"); break;
  case Jump::jg: assert(res == FE_CC_G && "FeCond value mismatch?"); break;
  case Jump::jge: assert(res == FE_CC_GE && "FeCond value mismatch?"); break;
  case Jump::jl: assert(res == FE_CC_L && "FeCond value mismatch?"); break;
  case Jump::jle: assert(res == FE_CC_LE && "FeCond value mismatch?"); break;
  case Jump::jne: assert(res == FE_CC_NE && "FeCond value mismatch?"); break;
  case Jump::jno: assert(res == FE_CC_NO && "FeCond value mismatch?"); break;
  case Jump::jo: assert(res == FE_CC_O && "FeCond value mismatch?"); break;
  case Jump::js: assert(res == FE_CC_S && "FeCond value mismatch?"); break;
  case Jump::jns: assert(res == FE_CC_NS && "FeCond value mismatch?"); break;
  case Jump::jp: assert(res == FE_CC_P && "FeCond value mismatch?"); break;
  case Jump::jnp: assert(res == FE_CC_NP && "FeCond value mismatch?"); break;
  default: TPDE_UNREACHABLE("invalid conditional jump");
  }
  return res;
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
void CompilerX64<Adaptor, Derived, BaseTy, Config>::generate_branch_to_block(
    const Jump jmp,
    IRBlockRef target,
    const bool needs_split,
    const bool last_inst) noexcept {
  const auto target_idx = this->analyzer.block_idx(target);
  if (!needs_split || jmp == Jump::jmp) {
    this->derived()->move_to_phi_nodes(target_idx);

    if (!last_inst || this->analyzer.block_idx(target) != this->next_block()) {
      generate_raw_jump(jmp, this->block_labels[(u32)target_idx]);
    }
  } else {
    auto tmp_label = this->text_writer.label_create();
    generate_raw_jump(invert_jump(jmp), tmp_label);

    this->derived()->move_to_phi_nodes(target_idx);

    generate_raw_jump(Jump::jmp, this->block_labels[(u32)target_idx]);

    this->label_place(tmp_label);
  }
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
void CompilerX64<Adaptor, Derived, BaseTy, Config>::generate_raw_jump(
    Jump jmp, Label target_label) noexcept {
  this->text_writer.ensure_space(6); // For safe ptr arithmetic on code buffer.
  bool pending = this->text_writer.label_is_pending(target_label);
  void *target = this->text_writer.cur_ptr();
  if (!pending) {
    target = this->text_writer.begin_ptr() +
             this->text_writer.label_offset(target_label);
  }

  if (jmp == Jump::jmp) {
    ASMNCF(JMP, pending ? FE_JMPL : 0, target);
  } else {
    ASMNCF(Jcc, (pending ? FE_JMPL : 0) | jump_to_cond(jmp), target);
  }

  if (pending) {
    this->text_writer.label_ref(target_label,
                                this->text_writer.offset() - 4,
                                LabelFixupKind::X64_JMP_OR_MEM_DISP);
  }
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> class BaseTy,
          typename Config>
void CompilerX64<Adaptor, Derived, BaseTy, Config>::generate_raw_set(
    Jump cc, AsmReg dst, bool zext) noexcept {
  if (zext) {
    ASM(MOV32ri, dst, 0);
  }
  ASMF(SETcc8r, jump_to_cond(cc), dst);
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> class BaseTy,
          typename Config>
void CompilerX64<Adaptor, Derived, BaseTy, Config>::generate_raw_mask(
    Jump cc, AsmReg dst) noexcept {
  // TODO: use sbb dst,dst/adc dest,-1 for carry flag
  generate_raw_set(cc, dst);
  ASM(NEG64r, dst);
}
template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> class BaseTy,
          typename Config>
void CompilerX64<Adaptor, Derived, BaseTy, Config>::generate_raw_cmov(
    Jump cc, AsmReg dst, AsmReg src, bool is_64) noexcept {
  if (is_64) {
    ASMF(CMOVcc64rr, jump_to_cond(cc), dst, src);
  } else {
    ASMF(CMOVcc32rr, jump_to_cond(cc), dst, src);
  }
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> class BaseTy,
          typename Config>
void CompilerX64<Adaptor, Derived, BaseTy, Config>::generate_raw_intext(
    AsmReg dst, AsmReg src, bool sign, u32 from, u32 to) noexcept {
  assert(from < to && to <= 64);
  assert(may_clobber_flags());
  if (!sign) {
    switch (from) {
    case 8: ASM(MOVZXr32r8, dst, src); break;
    case 16: ASM(MOVZXr32r16, dst, src); break;
    case 32: ASM(MOV32rr, dst, src); break;
    default:
      if (from < 32) {
        if (dst != src) {
          ASM(MOV32rr, dst, src);
        }
        ASM(AND32ri, dst, (uint32_t{1} << from) - 1);
      } else if (dst != src) {
        ASM(MOV64ri, dst, (uint64_t{1} << from) - 1);
        ASM(AND64rr, dst, src);
      } else {
        ScratchReg tmp{this};
        AsmReg tmp_reg = tmp.alloc_gp();
        ASM(MOV64ri, tmp_reg, (uint64_t{1} << from) - 1);
        ASM(AND64rr, dst, tmp_reg);
      }
    }
  } else if (to <= 32) {
    switch (from) {
    case 8: ASM(MOVSXr32r8, dst, src); break;
    case 16: ASM(MOVSXr32r16, dst, src); break;
    default:
      if (dst != src) {
        ASM(MOV32rr, dst, src);
      }
      ASM(SHL32ri, dst, 32 - from);
      ASM(SAR32ri, dst, 32 - from);
    }
  } else {
    switch (from) {
    case 8: ASM(MOVSXr64r8, dst, src); break;
    case 16: ASM(MOVSXr64r16, dst, src); break;
    case 32: ASM(MOVSXr64r32, dst, src); break;
    default:
      if (dst != src) {
        ASM(MOV64rr, dst, src);
      }
      ASM(SHL64ri, dst, 64 - from);
      ASM(SAR64ri, dst, 64 - from);
    }
  }
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> class BaseTy,
          typename Config>
void CompilerX64<Adaptor, Derived, BaseTy, Config>::generate_raw_bfi(
    AsmReg dst, AsmReg src, u32 lsb, u32 width) noexcept {
  assert(lsb < 63 && width < 64 && lsb + width <= 64 && width != 0);
  assert(may_clobber_flags());
  ScratchReg tmp1{this};
  AsmReg tmp1_reg = tmp1.alloc_gp();
  // First, clear relevant bits in dst.
  if (width == 1) {
    ASM(BTR64ri, dst, lsb);
  } else if (lsb + width <= 31) {
    ASM(AND64ri, dst, ~(((u64{1} << width) - 1) << lsb));
  } else {
    ASM(MOV64ri, tmp1_reg, ~(((u64{1} << width) - 1) << lsb));
    ASM(AND64rr, dst, tmp1_reg);
  }
  // Second, clear irrelevant bits in src; result is in tmp1_reg.
  if (width == 8) {
    ASM(MOVZXr32r8, tmp1_reg, src);
  } else if (width == 16) {
    ASM(MOVZXr32r16, tmp1_reg, src);
  } else if (width <= 32) {
    ASM(MOV32rr, tmp1_reg, src);
    if (width < 32) {
      ASM(AND32ri, tmp1_reg, (u32{1} << width) - 1);
    }
  } else {
    ASM(MOV64ri, tmp1_reg, (u64{1} << width) - 1);
    ASM(AND64rr, tmp1_reg, src);
  }
  // Third, merge. Bits are disjoint, so addition is possible.
  if (lsb >= 1 && lsb <= 3) {
    ASM(LEA64rm, dst, FE_MEM(dst, u8(1 << lsb), tmp1_reg, 0));
  } else {
    if (lsb > 0 && lsb + width <= 32) {
      ASM(SHL32ri, tmp1_reg, lsb);
    } else if (lsb > 0) {
      ASM(SHL64ri, tmp1_reg, lsb);
    }
    ASM(OR64rr, dst, tmp1_reg);
  }
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> class BaseTy,
          typename Config>
void CompilerX64<Adaptor, Derived, BaseTy, Config>::generate_raw_bfiz(
    AsmReg dst, AsmReg src, u32 lsb, u32 width) noexcept {
  assert(lsb < 63 && width < 64 && lsb + width <= 64 && width != 0);
  assert(dst != src);
  assert(may_clobber_flags());
  // Clear irrelevant bits in src and move to dst.
  if (width == 8) {
    ASM(MOVZXr32r8, dst, src);
  } else if (width == 16) {
    ASM(MOVZXr32r16, dst, src);
  } else if (width <= 32) {
    ASM(MOV32rr, dst, src);
    if (width < 32) {
      ASM(AND32ri, dst, (u32{1} << width) - 1);
    }
  } else {
    ASM(MOV64ri, dst, (u64{1} << width) - 1);
    ASM(AND64rr, dst, src);
  }
  // Shift into place.
  if (lsb > 0 && lsb + width <= 32) {
    ASM(SHL32ri, dst, lsb);
  } else if (lsb > 0) {
    ASM(SHL64ri, dst, lsb);
  }
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> class BaseTy,
          typename Config>
void CompilerX64<Adaptor, Derived, BaseTy, Config>::CallBuilder::
    set_stack_used() noexcept {
  if (this->compiler.stack.has_dynamic_alloca && stack_adjust_off == 0) {
    stack_adjust_off = this->compiler.text_writer.offset();
    // Always use 32-bit immediate
    ASMC(&this->compiler, SUB64ri, FE_SP, 0x100);
    assert(this->compiler.text_writer.offset() == stack_adjust_off + 7);
  }
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> class BaseTy,
          typename Config>
void CompilerX64<Adaptor, Derived, BaseTy, Config>::CallBuilder::add_arg_byval(
    ValuePart &vp, CCAssignment &cca) noexcept {
  AsmReg ptr = vp.load_to_reg(&this->compiler);
  ScratchReg scratch{&this->compiler};
  AsmReg tmp = scratch.alloc_gp();

  auto size = cca.size;
  set_stack_used();
  i32 off = 0;
  while (size >= 8) {
    ASMC(&this->compiler, MOV64rm, tmp, FE_MEM(ptr, 0, FE_NOREG, off));
    ASMC(&this->compiler,
         MOV64mr,
         FE_MEM(FE_SP, 0, FE_NOREG, (i32)(cca.stack_off + off)),
         tmp);
    off += 8;
    size -= 8;
  }
  if (size >= 4) {
    ASMC(&this->compiler, MOV32rm, tmp, FE_MEM(ptr, 0, FE_NOREG, off));
    ASMC(&this->compiler,
         MOV32mr,
         FE_MEM(FE_SP, 0, FE_NOREG, (i32)(cca.stack_off + off)),
         tmp);
    off += 4;
    size -= 4;
  }
  if (size >= 2) {
    ASMC(&this->compiler, MOVZXr32m16, tmp, FE_MEM(ptr, 0, FE_NOREG, off));
    ASMC(&this->compiler,
         MOV16mr,
         FE_MEM(FE_SP, 0, FE_NOREG, (i32)(cca.stack_off + off)),
         tmp);
    off += 2;
    size -= 2;
  }
  if (size >= 1) {
    ASMC(&this->compiler, MOVZXr32m8, tmp, FE_MEM(ptr, 0, FE_NOREG, off));
    ASMC(&this->compiler,
         MOV8mr,
         FE_MEM(FE_SP, 0, FE_NOREG, (i32)(cca.stack_off + off)),
         tmp);
  }
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> class BaseTy,
          typename Config>
void CompilerX64<Adaptor, Derived, BaseTy, Config>::CallBuilder::add_arg_stack(
    ValuePart &vp, CCAssignment &cca) noexcept {
  set_stack_used();

  auto reg = vp.has_reg() ? vp.cur_reg() : vp.load_to_reg(&this->compiler);
  if (this->compiler.register_file.reg_bank(reg) == Config::GP_BANK) {
    switch (cca.size) {
    case 1:
      ASMC(&this->compiler,
           MOV8mr,
           FE_MEM(FE_SP, 0, FE_NOREG, i32(cca.stack_off)),
           reg);
      break;
    case 2:
      ASMC(&this->compiler,
           MOV16mr,
           FE_MEM(FE_SP, 0, FE_NOREG, i32(cca.stack_off)),
           reg);
      break;
    case 4:
      ASMC(&this->compiler,
           MOV32mr,
           FE_MEM(FE_SP, 0, FE_NOREG, i32(cca.stack_off)),
           reg);
      break;
    case 8:
      ASMC(&this->compiler,
           MOV64mr,
           FE_MEM(FE_SP, 0, FE_NOREG, i32(cca.stack_off)),
           reg);
      break;
    default: TPDE_UNREACHABLE("invalid GP reg size");
    }
  } else {
    assert(this->compiler.register_file.reg_bank(reg) == Config::FP_BANK);
    switch (cca.size) {
    case 4:
      ASMC(&this->compiler,
           SSE_MOVSSmr,
           FE_MEM(FE_SP, 0, FE_NOREG, i32(cca.stack_off)),
           reg);
      break;
    case 8:
      ASMC(&this->compiler,
           SSE_MOVSDmr,
           FE_MEM(FE_SP, 0, FE_NOREG, i32(cca.stack_off)),
           reg);
      break;
    case 16:
      ASMC(&this->compiler,
           SSE_MOVDQAmr,
           FE_MEM(FE_SP, 0, FE_NOREG, i32(cca.stack_off)),
           reg);
      break;
    default: TPDE_UNREACHABLE("invalid GP reg size");
    }
  }
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> class BaseTy,
          typename Config>
void CompilerX64<Adaptor, Derived, BaseTy, Config>::CallBuilder::call_impl(
    std::variant<SymRef, ValuePart> &&target) noexcept {
  if (this->assigner.is_vararg()) {
    if (this->compiler.register_file.is_used(Reg{AsmReg::AX})) {
      this->compiler.evict_reg(Reg{AsmReg::AX});
    }
    Reg next_xmm = this->compiler.register_file.find_first_free_excluding(
        Config::FP_BANK, 0);
    unsigned xmm_cnt = 8;
    if (next_xmm.valid() && next_xmm.id() - AsmReg::XMM0 < 8) {
      xmm_cnt = next_xmm.id() - AsmReg::XMM0;
    }
    if (xmm_cnt != 0) {
      ASMC(&this->compiler, MOV32ri, FE_AX, xmm_cnt);
    } else {
      ASMC(&this->compiler, XOR32rr, FE_AX, FE_AX);
    }
  }

  u32 sub = 0;
  if (stack_adjust_off != 0) {
    auto *inst_ptr = this->compiler.text_writer.begin_ptr() + stack_adjust_off;
    sub = util::align_up(this->assigner.get_stack_size(), 0x10);
    memcpy(inst_ptr + 3, &sub, sizeof(u32));
  } else {
    auto &max_stack_size = this->compiler.max_callee_stack_arg_size;
    max_stack_size = std::max(max_stack_size, this->assigner.get_stack_size());
  }

  if (auto *sym = std::get_if<SymRef>(&target)) {
    this->compiler.text_writer.ensure_space(16);
    ASMC(&this->compiler, CALL, this->compiler.text_writer.cur_ptr());
    this->compiler.reloc_text(
        *sym, R_X86_64_PLT32, this->compiler.text_writer.offset() - 4, -4);
  } else {
    ValuePart &tvp = std::get<ValuePart>(target);
    if (tvp.has_assignment() && !tvp.assignment().register_valid()) {
      assert(tvp.assignment().stack_valid());
      auto off = tvp.assignment().frame_off();
      ASMC(&this->compiler, CALLm, FE_MEM(FE_BP, 0, FE_NOREG, off));
    } else if (tvp.can_salvage()) {
      ASMC(&this->compiler, CALLr, tvp.salvage(&this->compiler));
    } else {
      assert(!this->compiler.register_file.is_used(Reg{AsmReg::R10}));
      AsmReg reg = tvp.reload_into_specific_fixed(&this->compiler, AsmReg::R10);
      ASMC(&this->compiler, CALLr, reg);
    }
    tvp.reset(&this->compiler);
  }

  if (stack_adjust_off != 0) {
    ASMC(&this->compiler, ADD64ri, FE_SP, sub);
  }
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
void CompilerX64<Adaptor, Derived, BaseTy, Config>::generate_call(
    std::variant<SymRef, ValuePart> &&target,
    std::span<CallArg> arguments,
    typename Base::ValueRef *result,
    const bool variable_args) {
  CCAssignerSysV assigner{variable_args};
  CallBuilder cb{*derived(), assigner};
  for (auto &arg : arguments) {
    cb.add_arg(std::move(arg));
  }
  cb.call(std::move(target));
  if (result) {
    cb.add_ret(*result);
  }
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
void CompilerX64<Adaptor, Derived, BaseTy, Config>::switch_emit_cmp(
    const AsmReg cmp_reg,
    const AsmReg tmp_reg,
    const u64 case_value,
    const bool width_is_32) noexcept {
  if (width_is_32) {
    ASM(CMP32ri, cmp_reg, case_value);
  } else {
    if ((i64)((i32)case_value) == (i64)case_value) {
      ASM(CMP64ri, cmp_reg, case_value);
    } else {
      this->materialize_constant(&case_value, Config::GP_BANK, 8, tmp_reg);
      ASM(CMP64rr, cmp_reg, tmp_reg);
    }
  }
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
void CompilerX64<Adaptor, Derived, BaseTy, Config>::switch_emit_cmpeq(
    const Label case_label,
    const AsmReg cmp_reg,
    const AsmReg tmp_reg,
    const u64 case_value,
    const bool width_is_32) noexcept {
  switch_emit_cmp(cmp_reg, tmp_reg, case_value, width_is_32);
  generate_raw_jump(Jump::je, case_label);
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
bool CompilerX64<Adaptor, Derived, BaseTy, Config>::switch_emit_jump_table(
    Label default_label,
    std::span<const Label> labels,
    AsmReg cmp_reg,
    AsmReg tmp_reg,
    u64 low_bound,
    u64 high_bound,
    bool width_is_32) noexcept {
  // NB: we must not evict any registers here.
  if (low_bound != 0) {
    switch_emit_cmp(cmp_reg, tmp_reg, low_bound, width_is_32);
    generate_raw_jump(Jump::jb, default_label);
  }
  switch_emit_cmp(cmp_reg, tmp_reg, high_bound, width_is_32);
  generate_raw_jump(Jump::ja, default_label);

  if (width_is_32) {
    // zero-extend cmp_reg since we use the full width
    ASM(MOV32rr, cmp_reg, cmp_reg);
  }

  if (low_bound != 0) {
    if (i32(low_bound) == i64(low_bound)) {
      ASM(SUB64ri, cmp_reg, low_bound);
    } else {
      this->materialize_constant(&low_bound, Config::GP_BANK, 8, tmp_reg);
      ASM(SUB64rr, cmp_reg, tmp_reg);
    }
  }

  Label jump_table = this->text_writer.label_create();
  ASM(LEA64rm, tmp_reg, FE_MEM(FE_IP, 0, FE_NOREG, -1));
  // we reuse the jump offset stuff since the patch procedure is the same
  this->text_writer.label_ref(jump_table,
                              this->text_writer.offset() - 4,
                              LabelFixupKind::X64_JMP_OR_MEM_DISP);
  // load the 4 byte displacement from the jump table
  ASM(MOVSXr64m32, cmp_reg, FE_MEM(tmp_reg, 4, cmp_reg, 0));
  ASM(ADD64rr, tmp_reg, cmp_reg);
  ASM(JMPr, tmp_reg);

  this->text_writer.align(4);
  this->text_writer.ensure_space(4 + 4 * labels.size());
  this->label_place(jump_table);
  const u32 table_off = this->text_writer.offset();
  for (u32 i = 0; i < labels.size(); i++) {
    if (this->text_writer.label_is_pending(labels[i])) {
      this->text_writer.label_ref(labels[i],
                                  this->text_writer.offset(),
                                  LabelFixupKind::X64_JUMP_TABLE);
      this->text_writer.write(table_off);
    } else {
      const auto label_off = this->text_writer.label_offset(labels[i]);
      this->text_writer.write((i32)label_off - (i32)table_off);
    }
  }
  return true;
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
void CompilerX64<Adaptor, Derived, BaseTy, Config>::switch_emit_binary_step(
    const Label case_label,
    const Label gt_label,
    const AsmReg cmp_reg,
    const AsmReg tmp_reg,
    const u64 case_value,
    const bool width_is_32) noexcept {
  switch_emit_cmpeq(case_label, cmp_reg, tmp_reg, case_value, width_is_32);
  generate_raw_jump(Jump::ja, gt_label);
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
CompilerX64<Adaptor, Derived, BaseTy, Config>::ScratchReg
    CompilerX64<Adaptor, Derived, BaseTy, Config>::tls_get_addr(
        SymRef sym, TLSModel model) noexcept {
  switch (model) {
  default: // TODO: implement optimized access for non-gd-model
  case TLSModel::GlobalDynamic: {
    // Generate function call to __tls_get_addr; on x86-64, this takes a single
    // parameter in rdi.
    assert(!this->stack.is_leaf_function);
    assert(may_clobber_flags());
    this->stack.generated_call = true;
    auto csr = CCAssignerSysV::Info.callee_saved_regs;
    for (auto reg : util::BitSetIterator<>{this->register_file.used & ~csr}) {
      this->evict_reg(Reg{reg});
    }
    ScratchReg arg{this};
    AsmReg arg_reg = arg.alloc_specific(AsmReg::DI);

    // Call sequence with extra prefixes for linker relaxation. Code sequence
    // taken from "ELF Handling For Thread-Local Storage".
    this->text_writer.ensure_space(0x10);
    *this->text_writer.cur_ptr()++ = 0x66;
    ASMNC(LEA64rm, arg_reg, FE_MEM(FE_IP, 0, FE_NOREG, 0));
    this->reloc_text(sym, R_X86_64_TLSGD, this->text_writer.offset() - 4, -4);
    *this->text_writer.cur_ptr()++ = 0x66;
    *this->text_writer.cur_ptr()++ = 0x66;
    *this->text_writer.cur_ptr()++ = 0x48;
    ASMNC(CALL, this->text_writer.cur_ptr());
    if (!this->sym_tls_get_addr.valid()) [[unlikely]] {
      this->sym_tls_get_addr = this->assembler.sym_add_undef(
          "__tls_get_addr", Assembler::SymBinding::GLOBAL);
    }
    this->reloc_text(this->sym_tls_get_addr,
                     R_X86_64_PLT32,
                     this->text_writer.offset() - 4,
                     -4);
    arg.reset();

    ScratchReg res{this};
    res.alloc_specific(AsmReg::AX);
    return res;
  }
  }
}

} // namespace tpde::x64
