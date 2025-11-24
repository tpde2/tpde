// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include "tpde/AssemblerElf.hpp"
#include "tpde/AssignmentPartRef.hpp"
#include "tpde/CompilerBase.hpp"
#include "tpde/DWARF.hpp"
#include "tpde/ELF.hpp"
#include "tpde/arm64/FunctionWriterA64.hpp"
#include "tpde/base.hpp"
#include "tpde/util/SmallVector.hpp"
#include "tpde/util/misc.hpp"

#include <bit>
#include <disarm64.h>

// Helper macros for assembling in the compiler
#if defined(ASM) || defined(ASMNC) || defined(ASMC)
  #error Got definition for ASM macros from somewhere else. Maybe you included compilers for multiple architectures?
#endif

/// Encode an instruction with an explicit compiler pointer
#define ASMC(compiler, op, ...)                                                \
  ((compiler)->text_writer.write_inst(de64_##op(__VA_ARGS__)))
/// Encode an instruction into this
#define ASM(...) ASMC(this, __VA_ARGS__)
/// Encode an instruction without checking that enough space is available
#define ASMNC(op, ...)                                                         \
  (this->text_writer.write_inst_unchecked(de64_##op(__VA_ARGS__)))
/// Encode an instruction if the encoding is successful (returns true)
#define ASMIFC(compiler, op, ...)                                              \
  ((compiler)->text_writer.try_write_inst(de64_##op(__VA_ARGS__)))
/// Encode an instruction if the encoding is successful (returns true)
#define ASMIF(...) ASMIFC(this, __VA_ARGS__)

namespace tpde::a64 {

struct AsmReg : Reg {
  enum REG : u8 {
    R0 = 0,
    R1,
    R2,
    R3,
    R4,
    R5,
    R6,
    R7,
    R8,
    R9,
    R10,
    R11,
    R12,
    R13,
    R14,
    R15,
    R16,
    R17,
    R18,
    R19,
    R20,
    R21,
    R22,
    R23,
    R24,
    R25,
    R26,
    R27,
    R28,
    R29,
    FP = 29,
    R30,
    LR = 30,
    SP = 31,

    V0 = 32,
    V1,
    V2,
    V3,
    V4,
    V5,
    V6,
    V7,
    V8,
    V9,
    V10,
    V11,
    V12,
    V13,
    V14,
    V15,
    V16,
    V17,
    V18,
    V19,
    V20,
    V21,
    V22,
    V23,
    V24,
    V25,
    V26,
    V27,
    V28,
    V29,
    V30,
    V31
  };

  constexpr explicit AsmReg() noexcept : Reg((u8)0xFF) {}

  constexpr AsmReg(const REG id) noexcept : Reg((u8)id) {}

  constexpr AsmReg(const Reg base) noexcept : Reg(base) {}

  constexpr explicit AsmReg(const u8 id) noexcept : Reg(id) {
    assert(id <= SP || (id >= V0 && id <= V31));
  }

  constexpr explicit AsmReg(const u64 id) noexcept : Reg(id) {
    assert(id <= SP || (id >= V0 && id <= V31));
  }

  operator DA_GReg() const noexcept {
    assert(reg_id < V0);
    return DA_GReg{reg_id};
  }

  operator DA_GRegZR() const noexcept {
    assert(reg_id < V0);
    assert(reg_id != SP); // 31 means SP in our enums
    return DA_GRegZR{reg_id};
  }

  operator DA_GRegSP() const noexcept {
    assert(reg_id <= SP);
    return DA_GRegSP{reg_id};
  }

  operator DA_VReg() const noexcept {
    assert(reg_id >= V0 && reg_id <= V31);
    return DA_VReg{static_cast<u8>(reg_id - V0)};
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

/// AArch64 AAPCS calling convention.
class CCAssignerAAPCS : public CCAssigner {
  static constexpr CCInfo Info{
      // we reserve SP,FP,R16 and R17 for our special use cases
      .allocatable_regs =
          0xFFFF'FFFF'FFFF'FFFF &
          ~create_bitmask({AsmReg::SP, AsmReg::FP, AsmReg::R16, AsmReg::R17}),
      // callee-saved registers
      .callee_saved_regs = create_bitmask({
          AsmReg::R19,
          AsmReg::R20,
          AsmReg::R21,
          AsmReg::R22,
          AsmReg::R23,
          AsmReg::R24,
          AsmReg::R25,
          AsmReg::R26,
          AsmReg::R27,
          AsmReg::R28,
          AsmReg::V8,
          AsmReg::V9,
          AsmReg::V10,
          AsmReg::V11,
          AsmReg::V12,
          AsmReg::V13,
          AsmReg::V14,
          AsmReg::V15,
      }),
      .arg_regs = create_bitmask({
          AsmReg::R0,
          AsmReg::R1,
          AsmReg::R2,
          AsmReg::R3,
          AsmReg::R4,
          AsmReg::R5,
          AsmReg::R6,
          AsmReg::R7,
          AsmReg::R8, // sret register
          AsmReg::V0,
          AsmReg::V1,
          AsmReg::V2,
          AsmReg::V3,
          AsmReg::V4,
          AsmReg::V5,
          AsmReg::V6,
          AsmReg::V7,
      }),
  };

  // NGRN = Next General-purpose Register Number
  // NSRN = Next SIMD/FP Register Number
  // NSAA = Next Stack Argument Address
  u32 ngrn = 0, nsrn = 0, nsaa = 0;
  u32 ret_ngrn = 0, ret_nsrn = 0;

public:
  CCAssignerAAPCS() noexcept : CCAssigner(Info) {}

  void reset() noexcept override {
    ngrn = nsrn = nsaa = ret_ngrn = ret_nsrn = 0;
  }

  void assign_arg(CCAssignment &arg) noexcept override {
    if (arg.byval) [[unlikely]] {
      nsaa = util::align_up(nsaa, arg.align < 8 ? 8 : arg.align);
      arg.stack_off = nsaa;
      nsaa += arg.size;
      return;
    }

    if (arg.sret) [[unlikely]] {
      arg.reg = AsmReg{AsmReg::R8};
      return;
    }

    if (arg.bank == RegBank{0}) {
      if (arg.align > 8) {
        ngrn = util::align_up(ngrn, 2);
      }
      if (ngrn + arg.consecutive < 8) {
        arg.reg = Reg{AsmReg::R0 + ngrn};
        ngrn += 1;
      } else {
        ngrn = 8;
        nsaa = util::align_up(nsaa, arg.align < 8 ? 8 : arg.align);
        arg.stack_off = nsaa;
        nsaa += 8;
      }
    } else {
      if (nsrn + arg.consecutive < 8) {
        arg.reg = Reg{AsmReg::V0 + nsrn};
        nsrn += 1;
      } else {
        nsrn = 8;
        u32 size = util::align_up(arg.size, 8);
        nsaa = util::align_up(nsaa, size);
        arg.stack_off = nsaa;
        nsaa += size;
      }
    }
  }

  u32 get_stack_size() noexcept override { return nsaa; }

  void assign_ret(CCAssignment &arg) noexcept override {
    assert(!arg.byval && !arg.sret);
    if (arg.bank == RegBank{0}) {
      if (arg.align > 8) {
        ret_ngrn = util::align_up(ret_ngrn, 2);
      }
      if (ret_ngrn + arg.consecutive < 8) {
        arg.reg = Reg{AsmReg::R0 + ret_ngrn};
        ret_ngrn += 1;
      } else {
        assert(false);
      }
    } else {
      if (ret_nsrn + arg.consecutive < 8) {
        arg.reg = Reg{AsmReg::V0 + ret_nsrn};
        ret_nsrn += 1;
      } else {
        assert(false);
      }
    }
  }
};

struct PlatformConfig : CompilerConfigDefault {
  using Assembler = tpde::elf::AssemblerElfA64;
  using AsmReg = tpde::a64::AsmReg;
  using DefaultCCAssigner = CCAssignerAAPCS;
  using FunctionWriter = FunctionWriterA64;

  static constexpr RegBank GP_BANK{0};
  static constexpr RegBank FP_BANK{1};
  static constexpr bool FRAME_INDEXING_NEGATIVE = false;
  static constexpr u32 PLATFORM_POINTER_SIZE = 8;
  static constexpr u32 NUM_BANKS = 2;
};

/// Compiler mixin for targeting AArch64.
template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy =
              CompilerBase,
          typename Config = PlatformConfig>
struct CompilerA64 : BaseTy<Adaptor, Derived, Config> {
  using Base = BaseTy<Adaptor, Derived, Config>;

  using IRValueRef = typename Base::IRValueRef;
  using IRBlockRef = typename Base::IRBlockRef;
  using IRFuncRef = typename Base::IRFuncRef;

  using ScratchReg = typename Base::ScratchReg;
  using ValuePartRef = typename Base::ValuePartRef;
  using ValuePart = typename Base::ValuePart;
  using GenericValuePart = typename Base::GenericValuePart;

  using RegisterFile = typename Base::RegisterFile;

  using CallArg = typename Base::CallArg;

  using Base::derived;


  // TODO(ts): make this dependent on the number of callee-saved regs of the
  // current function or if there is a call in the function?
  static constexpr u32 NUM_FIXED_ASSIGNMENTS[PlatformConfig::NUM_BANKS] = {5,
                                                                           6};

  enum CPU_FEATURES : u32 {
    CPU_BASELINE = 0, // ARMV8.0
  };

  CPU_FEATURES cpu_feats = CPU_BASELINE;

  // When handling function arguments, we need to prevent argument registers
  // from being handed out as fixed registers
  //
  // Additionally, we prevent R0 and R1 from being fixed assignments to
  // prevent issues with exception handling
  u64 fixed_assignment_nonallocatable_mask =
      create_bitmask({AsmReg::R0, AsmReg::R1});
  u32 func_start_off = 0u, func_prologue_alloc = 0u;
  /// Offset to the `add sp, sp, XXX` instruction that the argument handling
  /// uses to access stack arguments if needed
  u32 func_arg_stack_add_off = ~0u;
  AsmReg func_arg_stack_add_reg = AsmReg::make_invalid();

  /// Permanent scratch register, e.g. to materialize constants/offsets. This is
  /// used by materialize_constant, load_from_stack, spill_reg.
  AsmReg permanent_scratch_reg = AsmReg::R16;

  u32 scalar_arg_count = 0xFFFF'FFFF, vec_arg_count = 0xFFFF'FFFF;
  u32 reg_save_frame_off = 0;
  util::SmallVector<u32, 8> func_ret_offs = {};

  /// Helper class for building call sequences.
  class CallBuilder : public Base::template CallBuilderBase<CallBuilder> {
    u32 stack_adjust_off = 0;
    u32 stack_size = 0;
    u32 stack_sub = 0;

    void set_stack_used() noexcept;

  public:
    /// Constructor.
    CallBuilder(Derived &compiler, CCAssigner &assigner) noexcept
        : Base::template CallBuilderBase<CallBuilder>(compiler, assigner) {}

    void add_arg_byval(ValuePart &vp, CCAssignment &cca) noexcept;
    void add_arg_stack(ValuePart &vp, CCAssignment &cca) noexcept;
    void call_impl(std::variant<SymRef, ValuePart> &&) noexcept;
    void reset_stack() noexcept;
  };

  // for now, always generate an object
  explicit CompilerA64(Adaptor *adaptor,
                       const CPU_FEATURES cpu_features = CPU_BASELINE)
      : Base{adaptor}, cpu_feats(cpu_features) {
    static_assert(std::is_base_of_v<CompilerA64, Derived>);
  }

  void start_func(u32) noexcept {}

  /// Begin prologue, prepare for assigning arguments.
  void prologue_begin(CCAssigner *cc_assigner) noexcept;
  /// Assign argument part. Returns the stack offset if the value should be
  /// initialized as stack variable.
  std::optional<i32> prologue_assign_arg_part(ValuePart &&vp,
                                              CCAssignment cca) noexcept;
  /// Finish prologue.
  void prologue_end(CCAssigner *cc_assigner) noexcept;

  // note: this has to call assembler->end_func
  void finish_func(u32 func_idx) noexcept;

  void reset() noexcept;

  // helpers

  void gen_func_epilog() noexcept;

  void
      spill_reg(const AsmReg reg, const u32 frame_off, const u32 size) noexcept;

  void load_from_stack(AsmReg dst,
                       i32 frame_off,
                       u32 size,
                       bool sign_extend = false) noexcept;

  void load_address_of_stack_var(AsmReg dst, AssignmentPartRef ap) noexcept;

  void mov(AsmReg dst, AsmReg src, u32 size) noexcept;

  GenericValuePart val_spill_slot(AssignmentPartRef ap) noexcept {
    assert(ap.stack_valid() && !ap.variable_ref());
    return typename GenericValuePart::Expr(AsmReg::R29, ap.frame_off());
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
  /// Materialize constant into a register.
  void materialize_constant(u64 const_u64,
                            RegBank bank,
                            u32 size,
                            AsmReg dst) noexcept {
    assert(size <= sizeof(const_u64));
    materialize_constant(&const_u64, bank, size, dst);
  }

  AsmReg select_fixed_assignment_reg(AssignmentPartRef, IRValueRef) noexcept;

  /// Jump conditions.
  struct Jump {
    // TDOO: naming consistency
    enum Kind : uint8_t {
      Jeq,       ///< Equal (Z == 1)
      Jne,       ///< Not equal (Z == 0)
      Jcs,       ///< Carry set (C == 1)
      Jhs = Jcs, ///< Unsigned higher or same (C == 1)
      Jcc,       ///< Carry clear (C == 0)
      Jlo = Jcc, ///< Unsigned lower (C == 0)
      Jmi,       ///< Minus, negative (N == 1)
      Jpl,       ///< Plus, positive or zero  (N == 0)
      Jvs,       ///< Overflow  (V == 1)
      Jvc,       ///< No Overflow (V == 0)
      Jhi,       ///< Unsigned higher (C == 1 && Z == 0)
      Jls,       ///< Unsigned lower or same (!(C == 1 && Z == 0))
      Jge,       ///< Signed greater than or equal (N == V)
      Jlt,       ///< Signed less than (N != V)
      Jgt,       ///< Signed greater than (Z == 0 && N == V)
      Jle,       ///< Signed lessthan or equal  (!(Z == 0 && N == V))
      jmp,       ///< Unconditional jump
      Cbz,       ///< Compare and branch if zero (Wn or Xn register)
      Cbnz,      ///< Compare and branch if not zero (Wn or Xn register)
      Tbz,       ///< Test single bit and branch if zero (Xn register)
      Tbnz,      ///< Test single bit and branch if not zero (Xn register)
    };

    Kind kind;
    AsmReg cmp_reg;
    bool cmp_is_32;
    u8 test_bit;

    /// Unconditional branch.
    constexpr Jump() : kind(Kind::jmp) {}

    /// Unconditional or conditional branch based on flags.
    constexpr Jump(Kind kind) : kind(kind), cmp_is_32(false), test_bit(0) {
      assert(kind != Cbz && kind != Cbnz && kind != Tbz && kind != Tbnz);
    }

    /// Cbz/Cbnz branch.
    constexpr Jump(Kind kind, AsmReg cmp_reg, bool cmp_is_32)
        : kind(kind), cmp_reg(cmp_reg), cmp_is_32(cmp_is_32), test_bit(0) {
      assert(kind == Cbz || kind == Cbnz);
    }

    /// Tbz/Tbnz branch.
    constexpr Jump(Kind kind, AsmReg cmp_reg, u8 test_bit)
        : kind(kind), cmp_reg(cmp_reg), cmp_is_32(false), test_bit(test_bit) {
      assert(kind == Tbz || kind == Tbnz);
    }

    constexpr Jump change_kind(Kind new_kind) const {
      auto cpy = *this;
      cpy.kind = new_kind;
      return cpy;
    }
  };

  Jump invert_jump(Jump jmp) noexcept;
  Jump swap_jump(Jump jmp) noexcept;

  /// Generate jump instruction to target label.
  void generate_raw_jump(Jump jmp, Label target) noexcept;

  /// Convert jump condition to disarms Da64Cond.
  /// \warning Cbz,Cbnz,Tbz and Tbnz are not supported
  Da64Cond jump_to_cond(Jump jmp) noexcept;
  /// Set dst to 1 if cc is true, otherwise set it to zero
  void generate_raw_set(Jump cc, AsmReg dst) noexcept;
  /// Set all bits of dst to 1 if cc is true, otherwise set dst to zero
  void generate_raw_mask(Jump cc, AsmReg dst) noexcept;

  /// Moves true_select into dst if cc is true,
  /// otherwise move false_select into dst
  void generate_raw_select(Jump cc,
                           AsmReg dst,
                           AsmReg true_select,
                           AsmReg false_select,
                           bool is_64) noexcept;

  /// Integer extension. src is not modified.
  void generate_raw_intext(
      AsmReg dst, AsmReg src, bool sign, u32 from, u32 to) noexcept;

  /// Bitfield insert. src is not modified.
  void generate_raw_bfi(AsmReg dst, AsmReg src, u32 lsb, u32 width) noexcept {
    ASM(BFIx, dst, src, lsb, width);
  }
  /// Bitfield insert in zero. src is not modified.
  void generate_raw_bfiz(AsmReg dst, AsmReg src, u32 lsb, u32 width) noexcept {
    ASM(UBFIZx, dst, src, lsb, width);
  }

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
void CompilerA64<Adaptor, Derived, BaseTy, Config>::CallBuilder::
    set_stack_used() noexcept {
  if (stack_adjust_off == 0) {
    this->compiler.text_writer.ensure_space(16);
    stack_adjust_off = this->compiler.text_writer.offset();
    this->compiler.text_writer.cur_ptr() += 4;
  }
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> class BaseTy,
          typename Config>
void CompilerA64<Adaptor, Derived, BaseTy, Config>::CallBuilder::add_arg_byval(
    ValuePart &vp, CCAssignment &cca) noexcept {
  AsmReg ptr_reg = vp.load_to_reg(&this->compiler);
  AsmReg tmp_reg = AsmReg::R16;

  auto size = cca.size;
  set_stack_used();
  for (u32 off = 0; off < size;) {
    if (size - off >= 8) {
      ASMC(&this->compiler, LDRxu, tmp_reg, ptr_reg, off);
      ASMC(&this->compiler, STRxu, tmp_reg, DA_SP, cca.stack_off + off);
      off += 8;
    } else if (size - off >= 4) {
      ASMC(&this->compiler, LDRwu, tmp_reg, ptr_reg, off);
      ASMC(&this->compiler, STRwu, tmp_reg, DA_SP, cca.stack_off + off);
      off += 4;
    } else if (size - off >= 2) {
      ASMC(&this->compiler, LDRHu, tmp_reg, ptr_reg, off);
      ASMC(&this->compiler, STRHu, tmp_reg, DA_SP, cca.stack_off + off);
      off += 2;
    } else {
      ASMC(&this->compiler, LDRBu, tmp_reg, ptr_reg, off);
      ASMC(&this->compiler, STRBu, tmp_reg, DA_SP, cca.stack_off + off);
      off += 1;
    }
  }
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> class BaseTy,
          typename Config>
void CompilerA64<Adaptor, Derived, BaseTy, Config>::CallBuilder::add_arg_stack(
    ValuePart &vp, CCAssignment &cca) noexcept {
  set_stack_used();

  auto reg = vp.has_reg() ? vp.cur_reg() : vp.load_to_reg(&this->compiler);
  if (this->compiler.register_file.reg_bank(reg) == Config::GP_BANK) {
    switch (cca.size) {
    case 1: ASMC(&this->compiler, STRBu, reg, DA_SP, cca.stack_off); break;
    case 2: ASMC(&this->compiler, STRHu, reg, DA_SP, cca.stack_off); break;
    case 4: ASMC(&this->compiler, STRwu, reg, DA_SP, cca.stack_off); break;
    case 8: ASMC(&this->compiler, STRxu, reg, DA_SP, cca.stack_off); break;
    default: TPDE_UNREACHABLE("invalid GP reg size");
    }
  } else {
    assert(this->compiler.register_file.reg_bank(reg) == Config::FP_BANK);
    switch (cca.size) {
    case 1: ASMC(&this->compiler, STRbu, reg, DA_SP, cca.stack_off); break;
    case 2: ASMC(&this->compiler, STRhu, reg, DA_SP, cca.stack_off); break;
    case 4: ASMC(&this->compiler, STRsu, reg, DA_SP, cca.stack_off); break;
    case 8: ASMC(&this->compiler, STRdu, reg, DA_SP, cca.stack_off); break;
    case 16: ASMC(&this->compiler, STRqu, reg, DA_SP, cca.stack_off); break;
    default: TPDE_UNREACHABLE("invalid FP reg size");
    }
  }
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> class BaseTy,
          typename Config>
void CompilerA64<Adaptor, Derived, BaseTy, Config>::CallBuilder::call_impl(
    std::variant<SymRef, ValuePart> &&target) noexcept {
  u32 sub = 0;
  if (stack_adjust_off != 0) {
    auto *text_data = this->compiler.text_writer.begin_ptr();
    u32 *write_ptr = reinterpret_cast<u32 *>(text_data + stack_adjust_off);
    u32 stack_size = this->assigner.get_stack_size();
    sub = util::align_up(stack_size, stack_size < 0x1000 ? 0x10 : 0x1000);
    *write_ptr = de64_SUBxi(DA_SP, DA_SP, sub);
  } else {
    assert(this->assigner.get_stack_size() == 0);
  }

  // For vector registers, only the lowest half is callee-saved. Evict all
  // value parts larger than 8 bytes.
  auto fp_regs = RegisterFile::bank_regs(Config::FP_BANK);
  auto fp_csrs = fp_regs & this->assigner.get_ccinfo().callee_saved_regs;
  auto used_fp_csrs = fp_csrs & this->compiler.register_file.used;
  for (auto reg_id : util::BitSetIterator<>{used_fp_csrs}) {
    Reg reg{reg_id};
    ValLocalIdx local_idx = this->compiler.register_file.reg_local_idx(reg);
    auto part = this->compiler.register_file.reg_part(reg);
    AssignmentPartRef ap{this->compiler.val_assignment(local_idx), part};
    if (ap.part_size() > 8) {
      this->compiler.evict(ap);
    }
  }

  if (auto *sym = std::get_if<SymRef>(&target)) {
    ASMC(&this->compiler, BL, 0);
    this->compiler.reloc_text(
        *sym, elf::R_AARCH64_CALL26, this->compiler.text_writer.offset() - 4);
  } else {
    ValuePart &tvp = std::get<ValuePart>(target);
    if (tvp.can_salvage()) {
      ASMC(&this->compiler, BLR, tvp.salvage(&this->compiler));
    } else {
      AsmReg reg = this->compiler.permanent_scratch_reg;
      tvp.reload_into_specific_fixed(&this->compiler, reg);
      ASMC(&this->compiler, BLR, reg);
    }
    tvp.reset(&this->compiler);
  }

  if (stack_adjust_off != 0) {
    ASMC(&this->compiler, ADDxi, DA_SP, DA_SP, sub);
  }
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
void CompilerA64<Adaptor, Derived, BaseTy, Config>::prologue_begin(
    CCAssigner *cc_assigner) noexcept {
  func_ret_offs.clear();
  func_start_off = this->text_writer.offset();

  const CCInfo &cc_info = cc_assigner->get_ccinfo();

  // We don't actually generate the prologue here and merely allocate space
  // for it. Right now, we don't know which callee-saved registers will be
  // used. While we could pad with nops, we later move the beginning of the
  // function so that small functions don't have to execute 9 nops.
  // See finish_func.
  this->stack.frame_size = 16; // FP, LR
  {
    auto csr = cc_info.callee_saved_regs;
    auto csr_gp = csr & this->register_file.bank_regs(Config::GP_BANK);
    auto csr_fp = csr & this->register_file.bank_regs(Config::FP_BANK);
    u32 gp_saves = std::popcount(csr_gp);
    u32 fp_saves = std::popcount(csr_fp);
    // LDP/STP can handle two registers of the same bank.
    u32 reg_save_size = 4 * ((gp_saves + 1) / 2 + (fp_saves + 1) / 2);
    // TODO: support CSR of Qx/Vx registers, not just Dx
    this->stack.frame_size += util::align_up(gp_saves * 8 + fp_saves * 8, 16);

    // Reserve space for sub sp, stp x29/x30, and mov x29, sp.
    func_prologue_alloc = reg_save_size + 12;
    this->text_writer.ensure_space(func_prologue_alloc);
    this->text_writer.cur_ptr() += func_prologue_alloc;
  }

  // TODO(ts): support larger stack alignments?

  if (this->adaptor->cur_is_vararg()) [[unlikely]] {
    this->stack.frame_used = true;
    reg_save_frame_off = this->stack.frame_size;
    // We additionally store a pointer to the stack area, which we can't compute
    // with a constant offset from the frame pointer. Add 16 bytes to maintain
    // alignment.
    this->stack.frame_size += 8 * 8 + 8 * 16 + 16;
    this->text_writer.ensure_space(4 * 8);
    ASMNC(STPx, DA_GP(0), DA_GP(1), DA_SP, reg_save_frame_off);
    ASMNC(STPx, DA_GP(2), DA_GP(3), DA_SP, reg_save_frame_off + 16);
    ASMNC(STPx, DA_GP(4), DA_GP(5), DA_SP, reg_save_frame_off + 32);
    ASMNC(STPx, DA_GP(6), DA_GP(7), DA_SP, reg_save_frame_off + 48);
    ASMNC(STPq, DA_V(0), DA_V(1), DA_SP, reg_save_frame_off + 64);
    ASMNC(STPq, DA_V(2), DA_V(3), DA_SP, reg_save_frame_off + 96);
    ASMNC(STPq, DA_V(4), DA_V(5), DA_SP, reg_save_frame_off + 128);
    ASMNC(STPq, DA_V(6), DA_V(7), DA_SP, reg_save_frame_off + 160);
  }

  this->func_arg_stack_add_off = ~0u;
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
std::optional<i32>
    CompilerA64<Adaptor, Derived, BaseTy, Config>::prologue_assign_arg_part(
        ValuePart &&vp, CCAssignment cca) noexcept {
  if (cca.reg.valid()) [[likely]] {
    vp.set_value_reg(this, cca.reg);
    // Mark register as allocatable as soon as it is assigned. If the argument
    // is unused, the register will be freed immediately and can be used for
    // later stack arguments.
    this->register_file.allocatable |= u64{1} << cca.reg.id();
    return {};
  }

  AsmReg dst = vp.alloc_reg(this);

  this->text_writer.ensure_space(8);
  AsmReg stack_reg = AsmReg::R17;
  // TODO: allocate an actual scratch register for this.
  assert(!(this->register_file.allocatable & (u64{1} << stack_reg.id())) &&
         "x17 must not be allocatable");
  if (this->func_arg_stack_add_off == ~0u) {
    this->func_arg_stack_add_off = this->text_writer.offset();
    this->func_arg_stack_add_reg = stack_reg;
    // Fixed in finish_func when frame size is known
    ASMNC(ADDxi, stack_reg, DA_SP, 0);
  }

  if (cca.byval) {
    ASMNC(ADDxi, dst, stack_reg, cca.stack_off);
  } else if (cca.bank == Config::GP_BANK) {
    switch (cca.size) {
    case 1: ASMNC(LDRBu, dst, stack_reg, cca.stack_off); break;
    case 2: ASMNC(LDRHu, dst, stack_reg, cca.stack_off); break;
    case 4: ASMNC(LDRwu, dst, stack_reg, cca.stack_off); break;
    case 8: ASMNC(LDRxu, dst, stack_reg, cca.stack_off); break;
    default: TPDE_UNREACHABLE("invalid GP reg size");
    }
  } else {
    assert(cca.bank == Config::FP_BANK);
    switch (cca.size) {
    case 1: ASMNC(LDRbu, dst, stack_reg, cca.stack_off); break;
    case 2: ASMNC(LDRhu, dst, stack_reg, cca.stack_off); break;
    case 4: ASMNC(LDRsu, dst, stack_reg, cca.stack_off); break;
    case 8: ASMNC(LDRdu, dst, stack_reg, cca.stack_off); break;
    case 16: ASMNC(LDRqu, dst, stack_reg, cca.stack_off); break;
    default: TPDE_UNREACHABLE("invalid FP reg size");
    }
  }
  return {};
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
void CompilerA64<Adaptor, Derived, BaseTy, Config>::prologue_end(
    CCAssigner *cc_assigner) noexcept {
  // Hack: we don't know the frame size, so for a va_start(), we cannot easily
  // compute the offset from the frame pointer. But we have a stack_reg here,
  // so use it for var args.
  if (this->adaptor->cur_is_vararg()) [[unlikely]] {
    this->stack.frame_used = true;
    AsmReg stack_reg = AsmReg::R17;
    // TODO: allocate an actual scratch register for this.
    assert(!(this->register_file.allocatable & (u64{1} << stack_reg.id())) &&
           "x17 must not be allocatable");
    if (this->func_arg_stack_add_off == ~0u) {
      this->func_arg_stack_add_off = this->text_writer.offset();
      this->func_arg_stack_add_reg = stack_reg;
      // Fixed in finish_func when frame size is known
      ASMC(this, ADDxi, stack_reg, DA_SP, 0);
    }
    ASM(ADDxi, stack_reg, stack_reg, cc_assigner->get_stack_size());
    ASM(STRxu, stack_reg, DA_GP(29), this->reg_save_frame_off + 192);

    // TODO: extract ngrn/nsrn from CCAssigner
    // TODO: this isn't quite accurate, e.g. for (i128, i128, i128, i64, i128),
    // this should be 8 but will end up with 7.
    const CCInfo &cc_info = cc_assigner->get_ccinfo();
    auto arg_regs = this->register_file.allocatable & cc_info.arg_regs;
    u32 ngrn = 8 - util::cnt_lz<u16>((arg_regs & 0xff) << 8 | 0x80);
    u32 nsrn = 8 - util::cnt_lz<u16>(((arg_regs >> 32) & 0xff) << 8 | 0x80);
    this->scalar_arg_count = ngrn;
    this->vec_arg_count = nsrn;
  }
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
void CompilerA64<Adaptor, Derived, BaseTy, Config>::finish_func(
    u32 func_idx) noexcept {
  auto csr = derived()->cur_cc_assigner()->get_ccinfo().callee_saved_regs;
  u64 saved_regs = this->register_file.clobbered & csr;

  auto stack_reg = DA_SP;
  if (this->stack.has_dynamic_alloca) {
    stack_reg = DA_GP(29);
  }

  auto final_frame_size = util::align_up(this->stack.frame_size, 16);
  if (final_frame_size > 4095) {
    // round up to 4k since SUB cannot encode immediates greater than 4095
    final_frame_size = util::align_up(final_frame_size, 4096);
    assert(final_frame_size < 16 * 1024 * 1024);
  }

  bool needs_stack_frame =
      this->stack.frame_used || this->stack.generated_call ||
      this->stack.has_dynamic_alloca || saved_regs != 0 ||
      (this->register_file.clobbered & (u64{1} << AsmReg::LR));

  this->text_writer.eh_begin_fde(this->get_personality_sym());

  u32 prologue_size = 0;
  if (needs_stack_frame) [[likely]] {
    // NB: code alignment factor 4, data alignment factor -8.
    util::SmallVector<u32, 16> prologue;
    // For small stack frames, remember the state at the very beginning, which
    // is identical to the state after the post-increment LDP. For large stack
    // frames, remember the state after the SP adjustment (encoding the
    // corresponding DW_def_cfa SP, framesize would be >=3 bytes; this way we
    // can get away with a DW_def_cfa_offset 0 after the ADD).
    if (!func_ret_offs.empty() && final_frame_size <= 0x1f8) {
      this->text_writer.eh_write_inst(dwarf::DW_CFA_remember_state);
    }
    this->text_writer.eh_write_inst(dwarf::DW_CFA_advance_loc, 1);
    this->text_writer.eh_write_inst(dwarf::DW_CFA_def_cfa_offset,
                                    final_frame_size);
    if (final_frame_size <= 0x1f8) {
      prologue.push_back(
          de64_STPx_pre(DA_GP(29), DA_GP(30), DA_SP, -int(final_frame_size)));
      prologue.push_back(de64_MOV_SPx(DA_GP(29), DA_SP));
    } else {
      if (!func_ret_offs.empty()) {
        this->text_writer.eh_write_inst(dwarf::DW_CFA_remember_state);
      }
      prologue.push_back(de64_SUBxi(DA_SP, DA_SP, final_frame_size));
      prologue.push_back(de64_STPx(DA_GP(29), DA_GP(30), DA_SP, 0));
      prologue.push_back(de64_MOV_SPx(DA_GP(29), DA_SP));
    }

    // Patched below
    auto fde_prologue_adv_off = this->text_writer.eh_writer.size();
    this->text_writer.eh_write_inst(dwarf::DW_CFA_advance_loc, 0);
    this->text_writer.eh_write_inst(dwarf::DW_CFA_def_cfa_register,
                                    dwarf::a64::DW_reg_fp);
    this->text_writer.eh_write_inst(
        dwarf::DW_CFA_offset, dwarf::a64::DW_reg_fp, final_frame_size / 8);
    this->text_writer.eh_write_inst(
        dwarf::DW_CFA_offset, dwarf::a64::DW_reg_lr, final_frame_size / 8 - 1);

    AsmReg last_reg = AsmReg::make_invalid();
    u32 frame_off = 16;
    for (auto reg : util::BitSetIterator{saved_regs}) {
      u8 dwarf_base = reg < 32 ? dwarf::a64::DW_reg_x0 : dwarf::a64::DW_reg_v0;
      u8 dwarf_reg = dwarf_base + reg % 32;
      u32 cfa_off = (final_frame_size - frame_off) / 8 - last_reg.valid();
      if ((dwarf_reg & dwarf::DWARF_CFI_PRIMARY_OPCODE_MASK) == 0) {
        this->text_writer.eh_write_inst(
            dwarf::DW_CFA_offset, dwarf_reg, cfa_off);
      } else {
        this->text_writer.eh_write_inst(
            dwarf::DW_CFA_offset_extended, dwarf_reg, cfa_off);
      }

      if (last_reg.valid()) {
        const auto reg_bank = this->register_file.reg_bank(AsmReg{reg});
        const auto last_bank = this->register_file.reg_bank(last_reg);
        if (reg_bank == last_bank) {
          if (reg_bank == Config::GP_BANK) {
            prologue.push_back(
                de64_STPx(last_reg, AsmReg{reg}, stack_reg, frame_off));
          } else {
            prologue.push_back(
                de64_STPd(last_reg, AsmReg{reg}, stack_reg, frame_off));
          }
          frame_off += 16;
          last_reg = AsmReg::make_invalid();
        } else {
          assert(last_bank == Config::GP_BANK && reg_bank == Config::FP_BANK);
          prologue.push_back(de64_STRxu(last_reg, stack_reg, frame_off));
          frame_off += 8;
          last_reg = AsmReg{reg};
        }
      } else {
        last_reg = AsmReg{reg};
      }
    }

    if (last_reg.valid()) {
      if (this->register_file.reg_bank(last_reg) == Config::GP_BANK) {
        prologue.push_back(de64_STRxu(last_reg, stack_reg, frame_off));
      } else {
        assert(this->register_file.reg_bank(last_reg) == Config::FP_BANK);
        prologue.push_back(de64_STRdu(last_reg, stack_reg, frame_off));
      }
    }

    assert(prologue.size() * sizeof(u32) <= func_prologue_alloc);

    assert(prologue.size() < 0x4c);
    this->text_writer.eh_writer.data()[fde_prologue_adv_off] =
        dwarf::DW_CFA_advance_loc | (prologue.size() - 1);

    std::memcpy(this->text_writer.begin_ptr() + func_start_off,
                prologue.data(),
                prologue.size() * sizeof(u32));

    prologue_size = prologue.size() * sizeof(u32);
  }

  if (func_arg_stack_add_off != ~0u) {
    auto *raw_inst_ptr = this->text_writer.begin_ptr() + func_arg_stack_add_off;
    u32 *inst_ptr = reinterpret_cast<u32 *>(raw_inst_ptr);
    if (needs_stack_frame) {
      *inst_ptr = de64_ADDxi(func_arg_stack_add_reg, DA_SP, final_frame_size);
    } else {
      *inst_ptr = de64_MOV_SPx(func_arg_stack_add_reg, DA_SP);
    }
  }

  if (!func_ret_offs.empty()) {
    u8 *text_data = this->text_writer.begin_ptr();
    if (func_ret_offs.back() == this->text_writer.offset() - 4) {
      this->text_writer.cur_ptr() -= 4;
      func_ret_offs.pop_back();
    }
    for (auto ret_off : func_ret_offs) {
      u32 *write_ptr = reinterpret_cast<u32 *>(text_data + ret_off);
      *write_ptr = de64_B((this->text_writer.offset() - ret_off) / 4);
    }

    // Epilogue mirrors prologue + RET
    this->text_writer.ensure_space(prologue_size + 4);

    if (this->stack.has_dynamic_alloca) {
      ASMNC(MOV_SPx, DA_SP, DA_GP(29));
    }

    AsmReg last_reg = AsmReg::make_invalid();
    u32 frame_off = 16;
    for (auto reg : util::BitSetIterator{saved_regs}) {
      if (last_reg.valid()) {
        const auto reg_bank = this->register_file.reg_bank(AsmReg{reg});
        const auto last_bank = this->register_file.reg_bank(last_reg);
        if (reg_bank == last_bank) {
          if (reg_bank == Config::GP_BANK) {
            ASMNC(LDPx, last_reg, AsmReg{reg}, stack_reg, frame_off);
          } else {
            ASMNC(LDPd, last_reg, AsmReg{reg}, stack_reg, frame_off);
          }
          frame_off += 16;
          last_reg = AsmReg::make_invalid();
        } else {
          assert(last_bank == Config::GP_BANK && reg_bank == Config::FP_BANK);
          ASMNC(LDRxu, last_reg, stack_reg, frame_off);
          frame_off += 8;
          last_reg = AsmReg{reg};
        }
        continue;
      }

      last_reg = AsmReg{reg};
    }

    if (last_reg.valid()) {
      if (this->register_file.reg_bank(last_reg) == Config::GP_BANK) {
        ASMNC(LDRxu, last_reg, stack_reg, frame_off);
      } else {
        ASMNC(LDRdu, last_reg, stack_reg, frame_off);
      }
    }
    if (needs_stack_frame) {
      u32 body_start = func_start_off + func_prologue_alloc;
      this->text_writer.eh_advance(this->text_writer.offset() - body_start + 4);
      this->text_writer.eh_write_inst(dwarf::DW_CFA_restore_state);
      if (final_frame_size <= 0x1f8) {
        ASMNC(LDPx_post, DA_GP(29), DA_GP(30), DA_SP, final_frame_size);
        // CFI is correct here.
      } else {
        ASMNC(LDPx, DA_GP(29), DA_GP(30), DA_SP, 0);
        // CFI is correct here, but we need to update the CFA after the ADD.
        ASMNC(ADDxi, DA_SP, DA_SP, final_frame_size);
        this->text_writer.eh_write_inst(dwarf::DW_CFA_advance_loc, 1);
        this->text_writer.eh_write_inst(dwarf::DW_CFA_def_cfa_offset, 0);
      }
    }

    ASMNC(RET, DA_GP(30));
  }

  // TODO(ts): honor cur_needs_unwind_info
  this->text_writer.remove_prologue_bytes(func_start_off + prologue_size,
                                          func_prologue_alloc - prologue_size);
  auto func_size = this->text_writer.offset() - func_start_off;
  auto func_sym = this->func_syms[func_idx];
  auto func_sec = this->text_writer.get_sec_ref();
  this->assembler.sym_def(func_sym, func_sec, func_start_off, func_size);
  this->text_writer.eh_end_fde();
  this->text_writer.except_encode_func();
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
void CompilerA64<Adaptor, Derived, BaseTy, Config>::reset() noexcept {
  func_ret_offs.clear();
  Base::reset();
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
void CompilerA64<Adaptor, Derived, BaseTy, Config>::gen_func_epilog() noexcept {
  // Patched at the end, just reserve the space here.
  func_ret_offs.push_back(this->text_writer.offset());
  this->text_writer.ensure_space(4); // Single branch to actual epilogue.
  this->text_writer.cur_ptr() += 4;
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
void CompilerA64<Adaptor, Derived, BaseTy, Config>::spill_reg(
    const AsmReg reg, const u32 frame_off, const u32 size) noexcept {
  assert(this->stack.frame_used);
  assert((size & (size - 1)) == 0);
  assert(util::align_up(frame_off, size) == frame_off);
  // We don't support stack frames that aren't encodeable with add/sub.
  assert(frame_off < 0x1'000'000);
  this->text_writer.ensure_space(8);

  u32 off = frame_off;
  auto addr_base = AsmReg{AsmReg::FP};
  if (off >= 0x1000 * size) [[unlikely]] {
    // We cannot encode the offset in the store instruction.
    ASMNC(ADDxi, permanent_scratch_reg, DA_GP(29), off & ~0xfff);
    off &= 0xfff;
    addr_base = permanent_scratch_reg;
  }

  assert(-static_cast<i32>(frame_off) < 0);
  if (reg.id() <= AsmReg::R30) {
    switch (size) {
    case 1: ASMNC(STRBu, reg, addr_base, off); break;
    case 2: ASMNC(STRHu, reg, addr_base, off); break;
    case 4: ASMNC(STRwu, reg, addr_base, off); break;
    case 8: ASMNC(STRxu, reg, addr_base, off); break;
    default: TPDE_UNREACHABLE("invalid register spill size");
    }
  } else {
    switch (size) {
    case 1: ASMNC(STRbu, reg, addr_base, off); break;
    case 2: ASMNC(STRhu, reg, addr_base, off); break;
    case 4: ASMNC(STRsu, reg, addr_base, off); break;
    case 8: ASMNC(STRdu, reg, addr_base, off); break;
    case 16: ASMNC(STRqu, reg, addr_base, off); break;
    default: TPDE_UNREACHABLE("invalid register spill size");
    }
  }
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
void CompilerA64<Adaptor, Derived, BaseTy, Config>::load_from_stack(
    const AsmReg dst,
    const i32 frame_off,
    const u32 size,
    const bool sign_extend) noexcept {
  assert(this->stack.frame_used);
  assert((size & (size - 1)) == 0);
  assert(util::align_up(frame_off, size) == frame_off);
  // We don't support stack frames that aren't encodeable with add/sub.
  assert(frame_off >= 0 && frame_off < 0x1'000'000);
  this->text_writer.ensure_space(8);

  u32 off = frame_off;
  auto addr_base = AsmReg{AsmReg::FP};
  if (off >= 0x1000 * size) [[unlikely]] {
    // need to calculate this explicitly
    addr_base = dst.id() <= AsmReg::R30 ? dst : permanent_scratch_reg;
    ASMNC(ADDxi, addr_base, DA_GP(29), off & ~0xfff);
    off &= 0xfff;
  }

  if (dst.id() <= AsmReg::R30) {
    if (!sign_extend) {
      switch (size) {
      case 1: ASMNC(LDRBu, dst, addr_base, off); break;
      case 2: ASMNC(LDRHu, dst, addr_base, off); break;
      case 4: ASMNC(LDRwu, dst, addr_base, off); break;
      case 8: ASMNC(LDRxu, dst, addr_base, off); break;
      default: TPDE_UNREACHABLE("invalid register spill size");
      }
    } else {
      switch (size) {
      case 1: ASMNC(LDRSBwu, dst, addr_base, off); break;
      case 2: ASMNC(LDRSHwu, dst, addr_base, off); break;
      case 4: ASMNC(LDRSWxu, dst, addr_base, off); break;
      case 8: ASMNC(LDRxu, dst, addr_base, off); break;
      default: TPDE_UNREACHABLE("invalid register spill size");
      }
    }
    return;
  }

  assert(!sign_extend);

  switch (size) {
  case 1: ASMNC(LDRbu, dst, addr_base, off); break;
  case 2: ASMNC(LDRhu, dst, addr_base, off); break;
  case 4: ASMNC(LDRsu, dst, addr_base, off); break;
  case 8: ASMNC(LDRdu, dst, addr_base, off); break;
  case 16: ASMNC(LDRqu, dst, addr_base, off); break;
  default: TPDE_UNREACHABLE("invalid register spill size");
  }
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
void CompilerA64<Adaptor, Derived, BaseTy, Config>::load_address_of_stack_var(
    const AsmReg dst, const AssignmentPartRef ap) noexcept {
  assert(this->stack.frame_used);
  auto frame_off = ap.variable_stack_off();
  assert(frame_off >= 0);
  if (!ASMIF(ADDxi, dst, DA_GP(29), frame_off)) {
    materialize_constant(frame_off, Config::GP_BANK, 4, dst);
    ASM(ADDx_uxtw, dst, DA_GP(29), dst, 0);
  }
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
void CompilerA64<Adaptor, Derived, BaseTy, Config>::mov(
    const AsmReg dst, const AsmReg src, const u32 size) noexcept {
  this->text_writer.ensure_space(4);
  assert(dst.valid());
  assert(src.valid());
  if (dst.id() <= AsmReg::SP && src.id() <= AsmReg::SP) {
    assert(dst.id() != AsmReg::SP && src.id() != AsmReg::SP);
    if (size > 4) {
      ASMNC(MOVx, dst, src);
    } else {
      ASMNC(MOVw, dst, src);
    }
  } else if (dst.id() >= AsmReg::V0 && src.id() >= AsmReg::V0) {
    ASMNC(ORR16b, dst, src, src);
  } else if (dst.id() <= AsmReg::SP) {
    assert(dst.id() != AsmReg::SP);
    // gp<-vector
    assert(src.id() >= AsmReg::V0);
    assert(size <= 8);
    if (size <= 4) {
      ASMNC(FMOVws, dst, src);
    } else {
      ASMNC(FMOVxd, dst, src);
    }
  } else {
    // vector<-gp
    assert(src.id() <= AsmReg::R30);
    assert(dst.id() >= AsmReg::V0);
    assert(size <= 8);
    if (size <= 4) {
      ASMNC(FMOVsw, dst, src);
    } else {
      ASMNC(FMOVdx, dst, src);
    }
  }
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
AsmReg CompilerA64<Adaptor, Derived, BaseTy, Config>::gval_expr_as_reg(
    GenericValuePart &gv) noexcept {
  auto &expr = std::get<typename GenericValuePart::Expr>(gv.state);

  ScratchReg scratch{derived()};
  if (!expr.has_base() && !expr.has_index()) {
    AsmReg dst = scratch.alloc_gp();
    derived()->materialize_constant(expr.disp, Config::GP_BANK, 8, dst);
    expr.disp = 0;
  } else if (!expr.has_base() && expr.has_index()) {
    AsmReg index_reg = expr.index_reg();
    if (std::holds_alternative<ScratchReg>(expr.index)) {
      scratch = std::move(std::get<ScratchReg>(expr.index));
    } else {
      (void)scratch.alloc_gp();
    }
    AsmReg dst = scratch.cur_reg();
    if ((expr.scale & (expr.scale - 1)) == 0) {
      const auto shift = util::cnt_tz<u64>(expr.scale);
      ASM(LSLxi, dst, index_reg, shift);
    } else {
      AsmReg tmp2 = permanent_scratch_reg;
      derived()->materialize_constant(expr.scale, Config::GP_BANK, 8, tmp2);
      ASM(MULx, dst, index_reg, tmp2);
    }
  } else if (expr.has_base() && expr.has_index()) {
    AsmReg base_reg = expr.base_reg();
    AsmReg index_reg = expr.index_reg();
    if (std::holds_alternative<ScratchReg>(expr.base)) {
      scratch = std::move(std::get<ScratchReg>(expr.base));
    } else if (std::holds_alternative<ScratchReg>(expr.index)) {
      scratch = std::move(std::get<ScratchReg>(expr.index));
    } else {
      (void)scratch.alloc_gp();
    }
    AsmReg dst = scratch.cur_reg();
    if ((expr.scale & (expr.scale - 1)) == 0) {
      const auto shift = util::cnt_tz<u64>(expr.scale);
      ASM(ADDx_lsl, dst, base_reg, index_reg, shift);
    } else {
      AsmReg tmp2 = permanent_scratch_reg;
      derived()->materialize_constant(expr.scale, Config::GP_BANK, 8, tmp2);
      ASM(MADDx, dst, index_reg, tmp2, base_reg);
    }
  } else if (expr.has_base() && !expr.has_index()) {
    AsmReg base_reg = expr.base_reg();
    if (std::holds_alternative<ScratchReg>(expr.base)) {
      scratch = std::move(std::get<ScratchReg>(expr.base));
    } else {
      (void)scratch.alloc_gp();
    }
    AsmReg dst = scratch.cur_reg();
    if (expr.disp != 0 && ASMIF(ADDxi, dst, base_reg, expr.disp)) {
      expr.disp = 0;
    } else if (dst != base_reg) {
      ASM(MOVx, dst, base_reg);
    }
  } else {
    TPDE_UNREACHABLE("inconsistent GenericValuePart::Expr");
  }

  AsmReg dst = scratch.cur_reg();
  if (expr.disp != 0) {
    if (!ASMIF(ADDxi, dst, dst, expr.disp)) {
      AsmReg tmp2 = permanent_scratch_reg;
      derived()->materialize_constant(expr.disp, Config::GP_BANK, 8, tmp2);
      ASM(ADDx, dst, dst, tmp2);
    }
  }

  gv.state = std::move(scratch);
  return dst;
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
void CompilerA64<Adaptor, Derived, BaseTy, Config>::alloca_fixed(
    u64 size, u32 align, ValuePart &res) noexcept {
  assert(this->stack.has_dynamic_alloca &&
         "function marked as not having dynamic allocas can't have alloca");
  assert(align != 0 && (align & (align - 1)) == 0 && "invalid alignment");
  size = tpde::util::align_up(size, 16);
  AsmReg res_reg = res.alloc_reg(this);
  if (size >= 0x10'0000) {
    auto tmp = permanent_scratch_reg;
    materialize_constant(size, Config::GP_BANK, 8, tmp);
    ASM(SUBx_uxtx, res_reg, DA_SP, tmp, 0);
  } else if (size >= 0x1000) {
    ASM(SUBxi, res_reg, DA_SP, size & 0xff'f000);
    ASM(SUBxi, res_reg, res_reg, size & 0xfff);
  } else {
    ASM(SUBxi, res_reg, DA_SP, size & 0xfff);
  }

  if (align > 16) {
    // The stack pointer is always at least 16-byte aligned.
    ASM(ANDxi, res_reg, res_reg, ~(u64{align} - 1));
  }

  if (size > 0) {
    ASM(MOV_SPx, DA_SP, res_reg);
  }
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
void CompilerA64<Adaptor, Derived, BaseTy, Config>::alloca_dynamic(
    u64 elem_size, ValuePart &&count, u32 align, ValuePart &res) noexcept {
  assert(this->stack.has_dynamic_alloca &&
         "function marked as not having dynamic allocas can't have alloca");
  assert(align != 0 && (align & (align - 1)) == 0 && "invalid alignment");
  AsmReg size_reg = count.has_reg() ? count.cur_reg() : count.load_to_reg(this);
  AsmReg res_reg = res.alloc_try_reuse(this, count);

  if (elem_size == 0) {
    ASM(MOVZw, res_reg, 0);
  } else if ((elem_size & (elem_size - 1)) == 0) {
    const auto shift = util::cnt_tz(elem_size);
    if (shift <= 4) {
      ASM(SUBx_uxtx, res_reg, DA_SP, size_reg, shift);
    } else {
      ASM(LSLxi, res_reg, size_reg, shift);
      ASM(SUBx_uxtx, res_reg, DA_SP, res_reg, 0);
    }
  } else {
    auto tmp = permanent_scratch_reg;
    materialize_constant(elem_size, Config::GP_BANK, 8, tmp);
    ASM(MULx, res_reg, size_reg, tmp);
    ASM(SUBx_uxtx, res_reg, DA_SP, res_reg, 0);
  }

  align = align > 16 ? align : 16;
  if (elem_size & (align - 1)) {
    ASM(ANDxi, res_reg, res_reg, ~(u64{align} - 1));
  }

  ASM(MOV_SPx, DA_SP, res_reg);
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
void CompilerA64<Adaptor, Derived, BaseTy, Config>::materialize_constant(
    const u64 *data, const RegBank bank, const u32 size, AsmReg dst) noexcept {
  this->text_writer.ensure_space(5 * 4);

  const auto const_u64 = data[0];
  if (bank == Config::GP_BANK) {
    assert(size <= 8);
    if (const_u64 == 0) {
      ASMNC(MOVZw, dst, 0);
      return;
    }

    this->text_writer.cur_ptr() +=
        sizeof(u32) *
        de64_MOVconst(reinterpret_cast<u32 *>(this->text_writer.cur_ptr()),
                      dst,
                      const_u64);
    return;
  }

  assert(bank == Config::FP_BANK);
  // Try instructions that take an immediate
  if (size == 4) {
    if (ASMIF(FMOVsi, dst, std::bit_cast<float>((u32)const_u64))) {
      return;
    } else if (ASMIF(MOVId, dst, static_cast<u32>(const_u64))) {
      return;
    }
  } else if (size == 8) {
    if (ASMIF(FMOVdi, dst, std::bit_cast<double>(const_u64))) {
      return;
    } else if (ASMIF(MOVId, dst, const_u64)) {
      return;
    }
  } else if (size == 16) {
    const auto high_u64 = data[1];
    if (const_u64 == high_u64 && ASMIF(MOVI2d, dst, const_u64)) {
      return;
    } else if (high_u64 == 0 && ASMIF(MOVId, dst, const_u64)) {
      return;
    }
  }

  // We must either load through a GP register of from memory. Both cases need a
  // GP register in the common case. We reserve x16/x17 for cases like this.
  if (size <= 16) {
    this->register_file.mark_clobbered(permanent_scratch_reg);
    // Copy from a GP register
    // TODO: always load from memory?
    if (size <= 8) {
      materialize_constant(data, Config::GP_BANK, size, permanent_scratch_reg);
      if (size <= 4) {
        ASMNC(FMOVsw, dst, permanent_scratch_reg);
      } else {
        ASMNC(FMOVdx, dst, permanent_scratch_reg);
      }
      return;
    }

    auto rodata = this->assembler.get_data_section(true, false);
    std::span<const u8> raw_data{reinterpret_cast<const u8 *>(data), size};
    auto sym = this->assembler.sym_def_data(
        rodata, "", raw_data, 16, Assembler::SymBinding::LOCAL);
    this->text_writer.ensure_space(8); // ensure contiguous instructions
    this->reloc_text(
        sym, elf::R_AARCH64_ADR_PREL_PG_HI21, this->text_writer.offset(), 0);
    ASMNC(ADRP, permanent_scratch_reg, 0, 0);
    this->reloc_text(
        sym, elf::R_AARCH64_LDST128_ABS_LO12_NC, this->text_writer.offset(), 0);
    ASMNC(LDRqu, dst, permanent_scratch_reg, 0);
    return;
  }

  TPDE_FATAL("unable to materialize constant");
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
AsmReg
    CompilerA64<Adaptor, Derived, BaseTy, Config>::select_fixed_assignment_reg(
        AssignmentPartRef ap, IRValueRef) noexcept {
  RegBank bank = ap.bank();
  if (bank == Config::FP_BANK && ap.part_size() > 8) {
    // FP registers can not in general be fixed registers, as only the lowest 8
    // bytes are callee-saved.
    return AsmReg::make_invalid();
  }

  // TODO(ts): why is this in here?
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

    assert(!this->register_file.is_fixed(reg));

    const auto local_idx = this->register_file.reg_local_idx(reg);
    const auto part = this->register_file.reg_part(reg);
    assert(local_idx != Base::INVALID_VAL_LOCAL_IDX);

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
          template <typename, typename, typename> class BaseTy,
          typename Config>
typename CompilerA64<Adaptor, Derived, BaseTy, Config>::Jump
    CompilerA64<Adaptor, Derived, BaseTy, Config>::invert_jump(
        Jump jmp) noexcept {
  switch (jmp.kind) {
  case Jump::Jeq: return jmp.change_kind(Jump::Jne);
  case Jump::Jne: return jmp.change_kind(Jump::Jeq);
  case Jump::Jcs: return jmp.change_kind(Jump::Jcc);
  case Jump::Jcc: return jmp.change_kind(Jump::Jcs);
  case Jump::Jmi: return jmp.change_kind(Jump::Jpl);
  case Jump::Jpl: return jmp.change_kind(Jump::Jmi);
  case Jump::Jvs: return jmp.change_kind(Jump::Jvc);
  case Jump::Jvc: return jmp.change_kind(Jump::Jvs);
  case Jump::Jhi: return jmp.change_kind(Jump::Jls);
  case Jump::Jls: return jmp.change_kind(Jump::Jhi);
  case Jump::Jge: return jmp.change_kind(Jump::Jlt);
  case Jump::Jlt: return jmp.change_kind(Jump::Jge);
  case Jump::Jgt: return jmp.change_kind(Jump::Jle);
  case Jump::Jle: return jmp.change_kind(Jump::Jgt);
  case Jump::jmp: return jmp;
  case Jump::Cbz: return jmp.change_kind(Jump::Cbnz);
  case Jump::Cbnz: return jmp.change_kind(Jump::Cbz);
  case Jump::Tbz: return jmp.change_kind(Jump::Tbnz);
  case Jump::Tbnz: return jmp.change_kind(Jump::Tbz);
  default: TPDE_UNREACHABLE("invalid jump kind");
  }
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
typename CompilerA64<Adaptor, Derived, BaseTy, Config>::Jump
    CompilerA64<Adaptor, Derived, BaseTy, Config>::swap_jump(
        Jump jmp) noexcept {
  switch (jmp.kind) {
  case Jump::Jeq: return jmp.change_kind(Jump::Jeq);
  case Jump::Jne: return jmp.change_kind(Jump::Jne);
  case Jump::Jcc: return jmp.change_kind(Jump::Jhi);
  case Jump::Jcs: return jmp.change_kind(Jump::Jls);
  case Jump::Jhi: return jmp.change_kind(Jump::Jcc);
  case Jump::Jls: return jmp.change_kind(Jump::Jcs);
  case Jump::Jge: return jmp.change_kind(Jump::Jle);
  case Jump::Jlt: return jmp.change_kind(Jump::Jgt);
  case Jump::Jgt: return jmp.change_kind(Jump::Jlt);
  case Jump::Jle: return jmp.change_kind(Jump::Jge);
  case Jump::jmp: return jmp;
  case Jump::Jmi:
  case Jump::Jpl:
  case Jump::Jvs:
  case Jump::Jvc:
  case Jump::Cbz:
  case Jump::Cbnz:
  case Jump::Tbz:
  case Jump::Tbnz:
  default: TPDE_UNREACHABLE("invalid jump kind for swap_jump");
  }
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
void CompilerA64<Adaptor, Derived, BaseTy, Config>::generate_raw_jump(
    Jump jmp, Label target_label) noexcept {
  const auto is_pending = this->text_writer.label_is_pending(target_label);
  this->text_writer.ensure_space(4);
  if (jmp.kind == Jump::jmp) {
    if (is_pending) {
      ASMNC(B, 0);
      this->text_writer.label_ref(target_label,
                                  this->text_writer.offset() - 4,
                                  LabelFixupKind::AARCH64_BR);
    } else {
      const auto label_off = this->text_writer.label_offset(target_label);
      const auto cur_off = this->text_writer.offset();
      assert(cur_off >= label_off);
      const auto diff = cur_off - label_off;
      assert((diff & 0b11) == 0);
      assert(diff < 128 * 1024 * 1024);

      ASMNC(B, -static_cast<ptrdiff_t>(diff) / 4);
    }
    return;
  }

  if (jmp.kind == Jump::Cbz || jmp.kind == Jump::Cbnz) {
    u32 off = 0;
    if (!is_pending) {
      const auto label_off = this->text_writer.label_offset(target_label);
      const auto cur_off = this->text_writer.offset();
      assert(cur_off >= label_off);
      off = cur_off - label_off;
      assert((off & 0b11) == 0);
      assert(off < 128 * 1024 * 1024);
    }

    if (off <= 1024 * 1024) {
      auto imm19 = -static_cast<ptrdiff_t>(off) / 4;
      if (jmp.kind == Jump::Cbz) {
        if (jmp.cmp_is_32) {
          ASMNC(CBZw, jmp.cmp_reg, imm19);
        } else {
          ASMNC(CBZx, jmp.cmp_reg, imm19);
        }
      } else {
        if (jmp.cmp_is_32) {
          ASMNC(CBNZw, jmp.cmp_reg, imm19);
        } else {
          ASMNC(CBNZx, jmp.cmp_reg, imm19);
        }
      }

      if (is_pending) {
        this->text_writer.label_ref(target_label,
                                    this->text_writer.offset() - 4,
                                    LabelFixupKind::AARCH64_COND_BR);
      }
    } else {
      assert(!is_pending);
      this->text_writer.ensure_space(2 * 4);

      if (jmp.kind == Jump::Cbz) {
        if (jmp.cmp_is_32) { // need to jump over 2 instructions
          ASMNC(CBNZw, jmp.cmp_reg, 2);
        } else {
          ASMNC(CBNZx, jmp.cmp_reg, 2);
        }
      } else {
        if (jmp.cmp_is_32) {
          ASMNC(CBZw, jmp.cmp_reg, 2);
        } else {
          ASMNC(CBZx, jmp.cmp_reg, 2);
        }
      }
      // + 4 since we already wrote the cb(n)z instruction
      ASMNC(B, -static_cast<ptrdiff_t>(off + 4) / 4);
    }
    return;
  }

  if (jmp.kind == Jump::Tbz || jmp.kind == Jump::Tbnz) {
    u32 off = 0;
    if (!is_pending) {
      const auto label_off = this->text_writer.label_offset(target_label);
      const auto cur_off = this->text_writer.offset();
      assert(cur_off >= label_off);
      off = cur_off - label_off;
      assert((off & 0b11) == 0);
      assert(off < 128 * 1024 * 1024);
    }

    if (off <= 32 * 1024) {
      auto imm14 = -static_cast<ptrdiff_t>(off) / 4;
      if (jmp.kind == Jump::Tbz) {
        ASMNC(TBZ, jmp.cmp_reg, jmp.test_bit, imm14);
      } else {
        ASMNC(TBNZ, jmp.cmp_reg, jmp.test_bit, imm14);
      }

      if (is_pending) {
        this->text_writer.label_ref(target_label,
                                    this->text_writer.offset() - 4,
                                    LabelFixupKind::AARCH64_TEST_BR);
      }
    } else {
      assert(!is_pending);
      this->text_writer.ensure_space(2 * 4);

      if (jmp.kind == Jump::Tbz) {
        // need to jump over 2 instructions
        ASMNC(TBNZ, jmp.cmp_reg, jmp.test_bit, 2);
      } else {
        ASMNC(TBZ, jmp.cmp_reg, jmp.test_bit, 2);
      }
      // + 4 since we already wrote the tb(n)z instruction
      ASMNC(B, -static_cast<ptrdiff_t>(off + 4) / 4);
    }
    return;
  }

  Da64Cond cond, cond_compl;
  switch (jmp.kind) {
  case Jump::Jeq:
    cond = DA_EQ;
    cond_compl = DA_NE;
    break;
  case Jump::Jne:
    cond = DA_NE;
    cond_compl = DA_EQ;
    break;
  case Jump::Jcs:
    cond = DA_CS;
    cond_compl = DA_CC;
    break;
  case Jump::Jcc:
    cond = DA_CC;
    cond_compl = DA_CS;
    break;
  case Jump::Jmi:
    cond = DA_MI;
    cond_compl = DA_PL;
    break;
  case Jump::Jpl:
    cond = DA_PL;
    cond_compl = DA_MI;
    break;
  case Jump::Jvs:
    cond = DA_VS;
    cond_compl = DA_VC;
    break;
  case Jump::Jvc:
    cond = DA_VC;
    cond_compl = DA_VS;
    break;
  case Jump::Jhi:
    cond = DA_HI;
    cond_compl = DA_LS;
    break;
  case Jump::Jls:
    cond = DA_LS;
    cond_compl = DA_HI;
    break;
  case Jump::Jge:
    cond = DA_GE;
    cond_compl = DA_LT;
    break;
  case Jump::Jlt:
    cond = DA_LT;
    cond_compl = DA_GE;
    break;
  case Jump::Jgt:
    cond = DA_GT;
    cond_compl = DA_LE;
    break;
  case Jump::Jle:
    cond = DA_LE;
    cond_compl = DA_GT;
    break;
  default: TPDE_UNREACHABLE("invalid jump kind");
  }


  u32 off = 0;
  if (!is_pending) {
    const auto label_off = this->text_writer.label_offset(target_label);
    const auto cur_off = this->text_writer.offset();
    assert(cur_off >= label_off);
    off = cur_off - label_off;
    assert((off & 0b11) == 0);
    assert(off < 128 * 1024 * 1024);
  }

  if (off <= 1024 * 1024) {
    ASMNC(BCOND, cond, -static_cast<ptrdiff_t>(off) / 4);

    if (is_pending) {
      this->text_writer.label_ref(target_label,
                                  this->text_writer.offset() - 4,
                                  LabelFixupKind::AARCH64_COND_BR);
    }
  } else {
    assert(!is_pending);
    this->text_writer.ensure_space(2 * 4);

    // 2 to skip over the branch following
    ASMNC(BCOND, cond_compl, 2);
    // + 4 since we already wrote the branch instruction
    ASMNC(B, -static_cast<ptrdiff_t>(off + 4) / 4);
  }
}
template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> class BaseTy,
          typename Config>
Da64Cond CompilerA64<Adaptor, Derived, BaseTy, Config>::jump_to_cond(
    Jump jmp) noexcept {
  switch (jmp.kind) {
  case Jump::Jeq: return DA_EQ;
  case Jump::Jne: return DA_NE;
  case Jump::Jcs: return DA_CS;
  case Jump::Jcc: return DA_CC;
  case Jump::Jmi: return DA_MI;
  case Jump::Jpl: return DA_PL;
  case Jump::Jvs: return DA_VS;
  case Jump::Jvc: return DA_VC;
  case Jump::Jhi: return DA_HI;
  case Jump::Jls: return DA_LS;
  case Jump::Jge: return DA_GE;
  case Jump::Jlt: return DA_LT;
  case Jump::Jgt: return DA_GT;
  case Jump::Jle: return DA_LE;
  case Jump::jmp: return DA_AL;
  default: TPDE_UNREACHABLE("invalid jump kind for conversion to Da64Cond");
  }
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> class BaseTy,
          typename Config>
void CompilerA64<Adaptor, Derived, BaseTy, Config>::generate_raw_set(
    Jump cc, AsmReg dst) noexcept {
  ASM(CSETw, dst, jump_to_cond(cc));
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> class BaseTy,
          typename Config>
void CompilerA64<Adaptor, Derived, BaseTy, Config>::generate_raw_mask(
    Jump cc, AsmReg dst) noexcept {
  ASM(CSETMx, dst, jump_to_cond(cc));
}
template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> class BaseTy,
          typename Config>
void CompilerA64<Adaptor, Derived, BaseTy, Config>::generate_raw_select(
    Jump cc,
    AsmReg dst,
    AsmReg true_select,
    AsmReg false_select,
    bool is_64) noexcept {
  this->text_writer.ensure_space(4);
  Da64Cond cond = jump_to_cond(cc);
  if (is_64) {
    ASMNC(CSELx, dst, true_select, false_select, cond);
  } else {
    ASMNC(CSELw, dst, true_select, false_select, cond);
  }
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> class BaseTy,
          typename Config>
void CompilerA64<Adaptor, Derived, BaseTy, Config>::generate_raw_intext(
    AsmReg dst, AsmReg src, bool sign, u32 from, u32 to) noexcept {
  assert(from < to && to <= 64);
  (void)to;
  if (sign) {
    if (to <= 32) {
      ASM(SBFXw, dst, src, 0, from);
    } else {
      ASM(SBFXx, dst, src, 0, from);
    }
  } else {
    if (to <= 32) {
      ASM(UBFXw, dst, src, 0, from);
    } else {
      ASM(UBFXx, dst, src, 0, from);
    }
  }
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
void CompilerA64<Adaptor, Derived, BaseTy, Config>::generate_call(
    std::variant<SymRef, ValuePart> &&target,
    std::span<CallArg> arguments,
    typename Base::ValueRef *result,
    bool) {
  CCAssignerAAPCS assigner;
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
void CompilerA64<Adaptor, Derived, BaseTy, Config>::switch_emit_cmp(
    AsmReg cmp_reg, AsmReg tmp_reg, u64 case_value, bool width_is_32) noexcept {
  if (width_is_32) {
    if (!ASMIF(CMPwi, cmp_reg, case_value)) {
      materialize_constant(case_value, Config::GP_BANK, 4, tmp_reg);
      ASM(CMPw, cmp_reg, tmp_reg);
    }
  } else {
    if (!ASMIF(CMPxi, cmp_reg, case_value)) {
      materialize_constant(case_value, Config::GP_BANK, 4, tmp_reg);
      ASM(CMPx, cmp_reg, tmp_reg);
    }
  }
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
void CompilerA64<Adaptor, Derived, BaseTy, Config>::switch_emit_cmpeq(
    Label case_label,
    AsmReg cmp_reg,
    AsmReg tmp_reg,
    u64 case_value,
    bool width_is_32) noexcept {
  switch_emit_cmp(cmp_reg, tmp_reg, case_value, width_is_32);
  generate_raw_jump(Jump::Jeq, case_label);
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
bool CompilerA64<Adaptor, Derived, BaseTy, Config>::switch_emit_jump_table(
    Label default_label,
    std::span<const Label> labels,
    AsmReg cmp_reg,
    AsmReg tmp_reg,
    u64 low_bound,
    u64 high_bound,
    bool width_is_32) noexcept {
  if (low_bound != 0) {
    switch_emit_cmp(cmp_reg, tmp_reg, low_bound, width_is_32);
    generate_raw_jump(Jump::Jcc, default_label);
  }
  switch_emit_cmp(cmp_reg, tmp_reg, high_bound, width_is_32);
  generate_raw_jump(Jump::Jhi, default_label);

  if (low_bound != 0) {
    if (!ASMIF(SUBxi, cmp_reg, cmp_reg, low_bound)) {
      this->materialize_constant(&low_bound, Config::GP_BANK, 8, tmp_reg);
      ASM(SUBx, cmp_reg, cmp_reg, tmp_reg);
    }
  }

  // TODO: move jump table to read-only data section.
  this->text_writer.ensure_space(4 * 4 + 4 * labels.size());

  Label jump_table = this->text_writer.label_create();
  u32 adr_off = this->text_writer.offset();
  this->text_writer.write_unchecked(u32(0)); // ADR tmp_reg, patched below.

  if (width_is_32) {
    ASMNC(LDRSWxr_uxtw, cmp_reg, tmp_reg, cmp_reg, /*scale=*/true);
  } else {
    ASMNC(LDRSWxr_lsl, cmp_reg, tmp_reg, cmp_reg, /*scale=*/true);
  }
  ASMNC(ADDx, tmp_reg, tmp_reg, cmp_reg);
  ASMNC(BR, tmp_reg);

  u32 table_off = this->text_writer.offset();
  this->text_writer.label_place(jump_table, table_off);
  for (Label label : labels) {
    this->text_writer.label_ref(
        label, this->text_writer.offset(), LabelFixupKind::AARCH64_JUMP_TABLE);
    this->text_writer.write_unchecked(table_off);
  }

  assert(table_off - adr_off <= 1 * 1024 * 1024); // ADR has a 1 MiB range.
  u32 *adr = reinterpret_cast<u32 *>(this->text_writer.begin_ptr() + adr_off);
  *adr = de64_ADR(tmp_reg, adr_off, table_off);
  return true;
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
void CompilerA64<Adaptor, Derived, BaseTy, Config>::switch_emit_binary_step(
    Label case_label,
    Label gt_label,
    AsmReg cmp_reg,
    AsmReg tmp_reg,
    u64 case_value,
    bool width_is_32) noexcept {
  switch_emit_cmpeq(case_label, cmp_reg, tmp_reg, case_value, width_is_32);
  generate_raw_jump(Jump::Jhi, gt_label);
}

template <IRAdaptor Adaptor,
          typename Derived,
          template <typename, typename, typename> typename BaseTy,
          typename Config>
CompilerA64<Adaptor, Derived, BaseTy, Config>::ScratchReg
    CompilerA64<Adaptor, Derived, BaseTy, Config>::tls_get_addr(
        SymRef sym, TLSModel model) noexcept {
  switch (model) {
  default: // TODO: implement optimized access for non-gd-model
  case TLSModel::GlobalDynamic: {
    assert(!this->stack.is_leaf_function);
    this->stack.generated_call = true;
    ScratchReg r0_scratch{this};
    AsmReg r0 = r0_scratch.alloc_specific(AsmReg::R0);
    ScratchReg r1_scratch{this};
    AsmReg r1 = r1_scratch.alloc_specific(AsmReg::R1);
    // The call only clobbers flags, x0, x1, and lr. x0 and x1 are already fixed
    // in the scratch registers, so only make sure that lr isn't used otherwise.
    if (this->register_file.is_used(Reg{AsmReg::LR})) {
      this->evict_reg(Reg{AsmReg::LR});
    }

    this->text_writer.ensure_space(0x18);
    this->reloc_text(
        sym, elf::R_AARCH64_TLSDESC_ADR_PAGE21, this->text_writer.offset(), 0);
    ASMNC(ADRP, r0, 0, 0);
    this->reloc_text(
        sym, elf::R_AARCH64_TLSDESC_LD64_LO12, this->text_writer.offset(), 0);
    ASMNC(LDRxu, r1, r0, 0);
    this->reloc_text(
        sym, elf::R_AARCH64_TLSDESC_ADD_LO12, this->text_writer.offset(), 0);
    ASMNC(ADDxi, r0, r0, 0);
    this->reloc_text(
        sym, elf::R_AARCH64_TLSDESC_CALL, this->text_writer.offset(), 0);
    ASMNC(BLR, r1);
    ASMNC(MRS, r1, 0xde82); // TPIDR_EL0
    // TODO: maybe return expr x0+x1.
    ASMNC(ADDx, r0, r1, r0);
    return r0_scratch;
  }
  }
}

} // namespace tpde::a64
