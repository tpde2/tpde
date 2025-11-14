// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include <algorithm>
#include <functional>
#include <unordered_map>
#include <variant>

#include "Analyzer.hpp"
#include "Compiler.hpp"
#include "CompilerConfig.hpp"
#include "IRAdaptor.hpp"
#include "tpde/AssignmentPartRef.hpp"
#include "tpde/FunctionWriter.hpp"
#include "tpde/RegisterFile.hpp"
#include "tpde/ValLocalIdx.hpp"
#include "tpde/ValueAssignment.hpp"
#include "tpde/base.hpp"
#include "tpde/util/function_ref.hpp"
#include "tpde/util/misc.hpp"

namespace tpde {
// TODO(ts): formulate concept for full compiler so that there is *some* check
// whether all the required derived methods are implemented?

/// Thread-local storage access mode
enum class TLSModel {
  GlobalDynamic,
  LocalDynamic,
  InitialExec,
  LocalExec,
};

struct CCAssignment {
  Reg reg = Reg::make_invalid(); ///< Assigned register, invalid implies stack.

  /// If non-zero, indicates that this and the next N values must be
  /// assigned to consecutive registers, or to the stack. The following
  /// values must be in the same register bank as this value. u8 is sufficient,
  /// no architecture has more than 255 parameter registers in a single bank.
  u8 consecutive = 0;
  bool sret : 1 = false; ///< Argument is return value pointer.

  /// The argument is passed by value on the stack. The provided argument is a
  /// pointer; for the call, size bytes will be copied into the corresponding
  /// stack slot. Behaves like LLVM's byval.
  ///
  /// Note: On x86-64 SysV, this is used to pass larger structs in memory. Note
  /// that AArch64 AAPCS doesn't use byval for structs, instead, the pointer is
  /// passed without byval and it is the responsibility of the caller to
  /// explicitly copy the value.
  bool byval : 1 = false;

  /// Extend integer argument. Highest bit indicates signed-ness, lower bits
  /// indicate the source width from which the argument should be extended.
  u8 int_ext = 0;

  u8 align = 0;             ///< Argument alignment
  RegBank bank = RegBank{}; ///< Register bank to assign the value to.
  u32 size = 0;             ///< Argument size, for byval the stack slot size.
  u32 stack_off = 0; ///< Assigned stack slot, only valid if reg is invalid.
};

struct CCInfo {
  // TODO: use RegBitSet
  const u64 allocatable_regs;
  const u64 callee_saved_regs;
  /// Possible argument registers; these registers will not be allocated until
  /// all arguments have been assigned.
  const u64 arg_regs;
  /// Size of red zone below stack pointer.
  const u8 red_zone_size = 0;
};

class CCAssigner {
public:
  const CCInfo *ccinfo;

  CCAssigner(const CCInfo &ccinfo) noexcept : ccinfo(&ccinfo) {}
  virtual ~CCAssigner() noexcept {}

  virtual void reset() noexcept = 0;

  const CCInfo &get_ccinfo() const noexcept { return *ccinfo; }

  virtual void assign_arg(CCAssignment &cca) noexcept = 0;
  virtual u32 get_stack_size() noexcept = 0;
  /// Some calling conventions need different call behavior when calling a
  /// vararg function.
  virtual bool is_vararg() const noexcept { return false; }
  virtual void assign_ret(CCAssignment &cca) noexcept = 0;
};

/// The base class for the compiler.
/// It implements the main platform independent compilation logic and houses the
/// analyzer
template <IRAdaptor Adaptor,
          typename Derived,
          CompilerConfig Config = CompilerConfigDefault>
struct CompilerBase {
  // some forwards for the IR type defs
  using IRValueRef = typename Adaptor::IRValueRef;
  using IRInstRef = typename Adaptor::IRInstRef;
  using IRBlockRef = typename Adaptor::IRBlockRef;
  using IRFuncRef = typename Adaptor::IRFuncRef;

  using BlockIndex = typename Analyzer<Adaptor>::BlockIndex;

  using AsmReg = typename Config::AsmReg;

  using RegisterFile = tpde::RegisterFile<Config::NUM_BANKS, 32>;

  /// A default implementation for ValRefSpecial.
  // Note: Subclasses can override this, always used Derived::ValRefSpecial.
  struct ValRefSpecial {
    uint8_t mode = 4;
    u64 const_data;
  };

#pragma region CompilerData
  Adaptor *adaptor;
  Analyzer<Adaptor> analyzer;

  // data for frame management

  struct {
    /// The current size of the stack frame
    u32 frame_size = 0;
    /// Whether the stack frame might have dynamic alloca. Dynamic allocas may
    /// require a different and less efficient frame setup.
    bool has_dynamic_alloca;
    /// Whether the function is guaranteed to be a leaf function. Throughout the
    /// entire function, the compiler may assume the absence of function calls.
    bool is_leaf_function;
    /// Whether the function actually includes a call. There are cases, where it
    /// is not clear from the beginning whether a function has function calls.
    /// If a function has no calls, this will allow using the red zone
    /// guaranteed by some ABIs.
    bool generated_call;
    /// Whether the stack frame is used. If the stack frame is never used, the
    /// frame setup can in some cases be omitted entirely.
    bool frame_used;
    /// Free-Lists for 1/2/4/8/16 sized allocations
    // TODO(ts): make the allocations for 4/8 different from the others
    // since they are probably the one's most used?
    util::SmallVector<i32, 16> fixed_free_lists[5] = {};
    /// Free-Lists for all other sizes
    // TODO(ts): think about which data structure we want here
    std::unordered_map<u32, std::vector<i32>> dynamic_free_lists{};
  } stack = {};

  typename Analyzer<Adaptor>::BlockIndex cur_block_idx;

  // Assignments

  static constexpr ValLocalIdx INVALID_VAL_LOCAL_IDX =
      static_cast<ValLocalIdx>(~0u);

  // TODO(ts): think about different ways to store this that are maybe more
  // compact?
  struct {
    AssignmentAllocator allocator;

    std::array<u32, Config::NUM_BANKS> cur_fixed_assignment_count = {};
    util::SmallVector<ValueAssignment *, Analyzer<Adaptor>::SMALL_VALUE_NUM>
        value_ptrs;

    ValLocalIdx variable_ref_list;
    util::SmallVector<ValLocalIdx, Analyzer<Adaptor>::SMALL_BLOCK_NUM>
        delayed_free_lists;
  } assignments = {};

  RegisterFile register_file;
#ifndef NDEBUG
  /// Whether we are currently in the middle of generating branch-related code
  /// and therefore must not change any value-related state.
  bool generating_branch = false;
#endif

private:
  /// Default CCAssigner if the implementation doesn't override cur_cc_assigner.
  typename Config::DefaultCCAssigner default_cc_assigner;

public:
  typename Config::Assembler assembler;
  Config::FunctionWriter text_writer;
  // TODO(ts): smallvector?
  std::vector<SymRef> func_syms;
  // TODO(ts): combine this with the block vectors in the analyzer to save on
  // allocations
  util::SmallVector<Label> block_labels;

  util::SmallVector<std::pair<SymRef, SymRef>, 4> personality_syms = {};

  struct ScratchReg;
  class ValuePart;
  struct ValuePartRef;
  struct ValueRef;
  struct GenericValuePart;
#pragma endregion

  struct InstRange {
    using Range = decltype(std::declval<Adaptor>().block_insts(
        std::declval<IRBlockRef>()));
    using Iter = decltype(std::declval<Range>().begin());
    using EndIter = decltype(std::declval<Range>().end());
    Iter from;
    EndIter to;
  };

  struct CallArg {
    enum class Flag : u8 {
      none,
      zext,
      sext,
      sret,
      byval,
      allow_split,
    };

    explicit CallArg(IRValueRef value,
                     Flag flags = Flag::none,
                     u8 byval_align = 1,
                     u32 byval_size = 0)
        : value(value),
          flag(flags),
          byval_align(byval_align),
          byval_size(byval_size) {}

    IRValueRef value;
    Flag flag;
    u8 byval_align;
    u8 ext_bits = 0;
    u32 byval_size;
  };

  template <typename CBDerived>
  class CallBuilderBase {
  protected:
    Derived &compiler;
    CCAssigner &assigner;

    RegisterFile::RegBitSet arg_regs{};

  public:
    CallBuilderBase(Derived &compiler, CCAssigner &assigner) noexcept
        : compiler(compiler), assigner(assigner) {}

    // CBDerived needs:
    // void add_arg_byval(ValuePart &vp, CCAssignment &cca) noexcept;
    // void add_arg_stack(ValuePart &vp, CCAssignment &cca) noexcept;
    // void call_impl(std::variant<SymRef, ValuePart> &&) noexcept;
    CBDerived *derived() noexcept { return static_cast<CBDerived *>(this); }

    void add_arg(ValuePart &&vp, CCAssignment cca) noexcept;
    void add_arg(const CallArg &arg, u32 part_count) noexcept;
    void add_arg(const CallArg &arg) noexcept {
      add_arg(std::move(arg), compiler.val_parts(arg.value).count());
    }

    // evict registers, do call, reset stack frame
    void call(std::variant<SymRef, ValuePart>) noexcept;

    void add_ret(ValuePart &vp, CCAssignment cca) noexcept;
    void add_ret(ValuePart &&vp, CCAssignment cca) noexcept {
      add_ret(vp, cca);
    }
    void add_ret(ValueRef &vr) noexcept;
  };

  class RetBuilder {
    Derived &compiler;
    CCAssigner &assigner;

    RegisterFile::RegBitSet ret_regs{};

  public:
    RetBuilder(Derived &compiler, CCAssigner &assigner) noexcept
        : compiler(compiler), assigner(assigner) {
      assigner.reset();
    }

    void add(ValuePart &&vp, CCAssignment cca) noexcept;
    void add(IRValueRef val) noexcept;

    void ret() noexcept;
  };

  /// Initialize a CompilerBase, should be called by the derived classes
  explicit CompilerBase(Adaptor *adaptor)
      : adaptor(adaptor), analyzer(adaptor), assembler() {
    static_assert(std::is_base_of_v<CompilerBase, Derived>);
    static_assert(Compiler<Derived, Config>);
  }

  /// shortcut for casting to the Derived class so that overloading
  /// works
  Derived *derived() { return static_cast<Derived *>(this); }

  const Derived *derived() const { return static_cast<const Derived *>(this); }

  [[nodiscard]] ValLocalIdx val_idx(const IRValueRef value) const noexcept {
    return analyzer.adaptor->val_local_idx(value);
  }

  [[nodiscard]] ValueAssignment *
      val_assignment(const ValLocalIdx idx) noexcept {
    return assignments.value_ptrs[static_cast<u32>(idx)];
  }

  /// Compile the functions returned by Adaptor::funcs
  ///
  /// \warning If you intend to call this multiple times, you must call reset
  ///   in-between the calls.
  ///
  /// \returns Whether the compilation was successful
  bool compile();

  /// Reset any leftover data from the previous compilation such that it will
  /// not affect the next compilation
  void reset();

  /// Get CCAssigner for current function.
  CCAssigner *cur_cc_assigner() noexcept { return &default_cc_assigner; }

  void init_assignment(IRValueRef value, ValLocalIdx local_idx) noexcept;

private:
  /// Frees an assignment, its stack slot and registers
  void free_assignment(ValLocalIdx local_idx, ValueAssignment *) noexcept;

public:
  /// Release an assignment when reference count drops to zero, either frees
  /// the assignment immediately or delays free to the end of the live range.
  void release_assignment(ValLocalIdx local_idx, ValueAssignment *) noexcept;

  /// Init a variable-ref assignment
  void init_variable_ref(ValLocalIdx local_idx, u32 var_ref_data) noexcept;
  /// Init a variable-ref assignment
  void init_variable_ref(IRValueRef value, u32 var_ref_data) noexcept {
    init_variable_ref(adaptor->val_local_idx(value), var_ref_data);
  }

  /// \name Stack Slots
  /// @{

  /// Allocate a static stack slot.
  i32 allocate_stack_slot(u32 size) noexcept;
  /// Free a static stack slot.
  void free_stack_slot(u32 slot, u32 size) noexcept;

  /// @}

  /// Assign function argument in prologue. \ref align can be used to increase
  /// the minimal stack alignment of the first part of the argument. If \ref
  /// allow_split is set, the argument can be passed partially in registers,
  /// otherwise (default) it must be either passed completely in registers or
  /// completely on the stack.
  void prologue_assign_arg(CCAssigner *cc_assigner,
                           u32 arg_idx,
                           IRValueRef arg,
                           u32 align = 1,
                           bool allow_split = false) noexcept;

  /// \name Value References
  /// @{

  /// Get a using reference to a value.
  ValueRef val_ref(IRValueRef value) noexcept;

  /// Get a using reference to a single-part value and provide direct access to
  /// the only part. This is a convenience function; note that the ValueRef must
  /// outlive the ValuePartRef (i.e. auto p = val_ref().part(0); won't work, as
  /// the value will possibly be deallocated when the ValueRef is destroyed).
  std::pair<ValueRef, ValuePartRef> val_ref_single(IRValueRef value) noexcept;

  /// Get a defining reference to a value.
  ValueRef result_ref(IRValueRef value) noexcept;

  /// Get a defining reference to a single-part value and provide direct access
  /// to the only part. Similar to val_ref_single().
  std::pair<ValueRef, ValuePartRef>
      result_ref_single(IRValueRef value) noexcept;

  /// Make dst an alias for src, which must be a non-constant value with an
  /// identical part configuration. src must be in its last use (is_owned()),
  /// and the assignment will be repurposed for dst, keeping all assigned
  /// registers and stack slots.
  ValueRef result_ref_alias(IRValueRef dst, ValueRef &&src) noexcept;

  /// Initialize value as a pointer into a stack variable (i.e., a value
  /// allocated from cur_static_allocas() or similar) with an offset. The
  /// result value will be a stack variable itself.
  ValueRef result_ref_stack_slot(IRValueRef value,
                                 AssignmentPartRef base,
                                 i32 off) noexcept;

  /// @}

  [[deprecated("Use ValuePartRef::set_value")]]
  void set_value(ValuePartRef &val_ref, ScratchReg &scratch) noexcept;
  [[deprecated("Use ValuePartRef::set_value")]]
  void set_value(ValuePartRef &&val_ref, ScratchReg &scratch) noexcept {
    set_value(val_ref, scratch);
  }

  /// Get generic value part into a single register, evaluating expressions
  /// and materializing immediates as required.
  AsmReg gval_as_reg(GenericValuePart &gv) noexcept;

  /// Like gval_as_reg; if the GenericValuePart owns a reusable register
  /// (either a ScratchReg, possibly due to materialization, or a reusable
  /// ValuePartRef), store it in dst.
  AsmReg gval_as_reg_reuse(GenericValuePart &gv, ScratchReg &dst) noexcept;

private:
  /// @internal Select register when a value needs to be evicted.
  Reg select_reg_evict(RegBank bank, u64 exclusion_mask) noexcept;

public:
  /// \name Low-Level Assignment Register Handling
  /// @{

  /// Select an available register, evicting loaded values if needed.
  Reg select_reg(RegBank bank, u64 exclusion_mask) noexcept {
    Reg res = register_file.find_first_free_excluding(bank, exclusion_mask);
    if (res.valid()) [[likely]] {
      return res;
    }
    return select_reg_evict(bank, exclusion_mask);
  }

  /// Reload a value part from memory or recompute variable address.
  void reload_to_reg(AsmReg dst, AssignmentPartRef ap) noexcept;

  /// Allocate a stack slot for an assignment.
  void allocate_spill_slot(AssignmentPartRef ap) noexcept;

  /// Ensure the value is spilled in its stack slot (except variable refs).
  void spill(AssignmentPartRef ap) noexcept;

  /// Evict the value from its register, spilling if needed, and free register.
  void evict(AssignmentPartRef ap) noexcept;

  /// Evict the value from the register, spilling if needed, and free register.
  void evict_reg(Reg reg) noexcept;

  /// Free the register. Requires that the contained value is already spilled.
  void free_reg(Reg reg) noexcept;

  /// @}

  /// \name High-Level Branch Generation
  /// @{

  /// Generate an unconditional branch at the end of a basic block. No further
  /// instructions must follow. If target is the next block in the block order,
  /// the branch is omitted.
  void generate_uncond_branch(IRBlockRef target) noexcept;

  /// Generate an conditional branch at the end of a basic block.
  template <typename Jump>
  void generate_cond_branch(Jump jmp,
                            IRBlockRef true_target,
                            IRBlockRef false_target) noexcept;

  /// Generate a switch at the end of a basic block. Only the lowest bits of the
  /// condition are considered. The condition must be a general-purpose
  /// register. The cases must be sorted and every case value must appear at
  /// most once.
  void generate_switch(
      ScratchReg &&cond,
      u32 width,
      IRBlockRef default_block,
      std::span<const std::pair<u64, IRBlockRef>> cases) noexcept;

  /// @}

  /// \name Low-Level Branch Primitives
  /// The general flow of using these low-level primitive is:
  /// 1. spill_before_branch()
  /// 2. begin_branch_region()
  /// 3. One or more calls to generate_branch_to_block()
  /// 4. end_branch_region()
  /// 5. release_spilled_regs()
  /// @{

  // TODO(ts): switch to a branch_spill_before naming style?
  /// Spill values that need to be spilled for later blocks. Returns the set
  /// of registers that will be free'd at the end of the block; pass this to
  /// release_spilled_regs().
  typename RegisterFile::RegBitSet
      spill_before_branch(bool force_spill = false) noexcept;
  /// Free registers marked by spill_before_branch().
  void release_spilled_regs(typename RegisterFile::RegBitSet) noexcept;

  /// When reaching a point in the function where no other blocks will be
  /// reached anymore, use this function to release register assignments after
  /// the end of that block so the compiler does not accidentally use
  /// registers which don't contain any values
  void release_regs_after_return() noexcept;

  /// Indicate beginning of region where value-state must not change.
  void begin_branch_region() noexcept {
#ifndef NDEBUG
    assert(!generating_branch);
    generating_branch = true;
#endif
  }

  /// Indicate end of region where value-state must not change.
  void end_branch_region() noexcept {
#ifndef NDEBUG
    assert(generating_branch);
    generating_branch = false;
#endif
  }

  /// Generate a branch to a basic block; execution continues afterwards.
  /// Multiple calls to this function can be used to build conditional branches.
  /// @tparam Jump Target-defined Jump type (e.g., CompilerX64::Jump).
  /// @param needs_split Result of branch_needs_split(); pass false for an
  ///  unconditional branch.
  /// @param last_inst Whether fall-through to target is possible.
  template <typename Jump>
  void generate_branch_to_block(Jump jmp,
                                IRBlockRef target,
                                bool needs_split,
                                bool last_inst) noexcept;

#ifndef NDEBUG
  bool may_change_value_state() const noexcept { return !generating_branch; }
#endif

  void move_to_phi_nodes(BlockIndex target) noexcept {
    if (analyzer.block_has_phis(target)) {
      move_to_phi_nodes_impl(target);
    }
  }

  void move_to_phi_nodes_impl(BlockIndex target) noexcept;

  /// Whether branch to a block requires additional instructions and therefore
  /// a direct jump to the block is not possible.
  bool branch_needs_split(IRBlockRef target) noexcept {
    // for now, if the target has PHI-nodes, we split
    return analyzer.block_has_phis(target);
  }

  /// @}

  BlockIndex next_block() const noexcept;

  bool try_force_fixed_assignment(IRValueRef) const noexcept { return false; }

  bool hook_post_func_sym_init() noexcept { return true; }

  void analysis_start() noexcept {}

  void analysis_end() noexcept {}

  void reloc_text(SymRef sym, u32 type, u64 offset, i64 addend = 0) noexcept {
    this->assembler.reloc_sec(
        text_writer.get_sec_ref(), sym, type, offset, addend);
  }

  /// Convenience function to place a label at the current position.
  void label_place(Label label) noexcept {
    this->text_writer.label_place(label, text_writer.offset());
  }

protected:
  SymRef get_personality_sym() noexcept;

  bool compile_func(IRFuncRef func, u32 func_idx) noexcept;

  bool compile_block(IRBlockRef block, u32 block_idx) noexcept;
};
} // namespace tpde

#include "GenericValuePart.hpp"
#include "ScratchReg.hpp"
#include "ValuePartRef.hpp"
#include "ValueRef.hpp"

namespace tpde {

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
template <typename CBDerived>
void CompilerBase<Adaptor, Derived, Config>::CallBuilderBase<
    CBDerived>::add_arg(ValuePart &&vp, CCAssignment cca) noexcept {
  if (!cca.byval && cca.bank == RegBank{}) {
    cca.bank = vp.bank();
    cca.size = vp.part_size();
  }

  assigner.assign_arg(cca);
  bool needs_ext = cca.int_ext != 0;
  bool ext_sign = cca.int_ext >> 7;
  unsigned ext_bits = cca.int_ext & 0x3f;

  if (cca.byval) {
    derived()->add_arg_byval(vp, cca);
    vp.reset(&compiler);
  } else if (!cca.reg.valid()) {
    if (needs_ext) {
      auto ext = std::move(vp).into_extended(&compiler, ext_sign, ext_bits, 64);
      derived()->add_arg_stack(ext, cca);
      ext.reset(&compiler);
    } else {
      derived()->add_arg_stack(vp, cca);
    }
    vp.reset(&compiler);
  } else {
    u32 size = vp.part_size();
    if (vp.is_in_reg(cca.reg)) {
      if (!vp.can_salvage()) {
        compiler.evict_reg(cca.reg);
      } else {
        vp.salvage(&compiler);
      }
      if (needs_ext) {
        compiler.generate_raw_intext(cca.reg, cca.reg, ext_sign, ext_bits, 64);
      }
    } else {
      if (compiler.register_file.is_used(cca.reg)) {
        compiler.evict_reg(cca.reg);
      }
      if (vp.can_salvage()) {
        AsmReg vp_reg = vp.salvage(&compiler);
        if (needs_ext) {
          compiler.generate_raw_intext(cca.reg, vp_reg, ext_sign, ext_bits, 64);
        } else {
          compiler.mov(cca.reg, vp_reg, size);
        }
      } else if (needs_ext && vp.is_const()) {
        u64 val = vp.const_data()[0];
        u64 extended =
            ext_sign ? util::sext(val, ext_bits) : util::zext(val, ext_bits);
        compiler.materialize_constant(&extended, cca.bank, 8, cca.reg);
      } else {
        vp.reload_into_specific_fixed(&compiler, cca.reg);
        if (needs_ext) {
          compiler.generate_raw_intext(
              cca.reg, cca.reg, ext_sign, ext_bits, 64);
        }
      }
    }
    vp.reset(&compiler);
    assert(!compiler.register_file.is_used(cca.reg));
    compiler.register_file.mark_clobbered(cca.reg);
    compiler.register_file.allocatable &= ~(u64{1} << cca.reg.id());
    arg_regs |= (1ull << cca.reg.id());
  }
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
template <typename CBDerived>
void CompilerBase<Adaptor, Derived, Config>::CallBuilderBase<
    CBDerived>::add_arg(const CallArg &arg, u32 part_count) noexcept {
  ValueRef vr = compiler.val_ref(arg.value);

  if (arg.flag == CallArg::Flag::byval) {
    assert(part_count == 1);
    add_arg(vr.part(0),
            CCAssignment{
                .byval = true,
                .align = arg.byval_align,
                .size = arg.byval_size,
            });
    return;
  }

  u32 align = arg.byval_align;
  bool allow_split = arg.flag == CallArg::Flag::allow_split;

  for (u32 part_idx = 0; part_idx < part_count; ++part_idx) {
    u8 int_ext = 0;
    if (arg.flag == CallArg::Flag::sext || arg.flag == CallArg::Flag::zext) {
      assert(arg.ext_bits != 0 && "cannot extend zero-bit integer");
      int_ext = arg.ext_bits | (arg.flag == CallArg::Flag::sext ? 0x80 : 0);
    }
    u32 remaining = part_count < 256 ? part_count - part_idx - 1 : 255;
    derived()->add_arg(vr.part(part_idx),
                       CCAssignment{
                           .consecutive = u8(allow_split ? 0 : remaining),
                           .sret = arg.flag == CallArg::Flag::sret,
                           .int_ext = int_ext,
                           .align = u8(part_idx == 0 ? align : 1),
                       });
  }
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
template <typename CBDerived>
void CompilerBase<Adaptor, Derived, Config>::CallBuilderBase<CBDerived>::call(
    std::variant<SymRef, ValuePart> target) noexcept {
  assert(!compiler.stack.is_leaf_function && "leaf func must not have calls");
  compiler.stack.generated_call = true;
  typename RegisterFile::RegBitSet skip_evict = arg_regs;
  if (auto *vp = std::get_if<ValuePart>(&target); vp && vp->can_salvage()) {
    // call_impl will reset vp, thereby unlock+free the register.
    assert(vp->cur_reg_unlocked().valid() && "can_salvage implies register");
    skip_evict |= (1ull << vp->cur_reg_unlocked().id());
  }

  auto clobbered = ~assigner.get_ccinfo().callee_saved_regs;
  for (auto reg_id : util::BitSetIterator<>{compiler.register_file.used &
                                            clobbered & ~skip_evict}) {
    compiler.evict_reg(AsmReg{reg_id});
    compiler.register_file.mark_clobbered(Reg{reg_id});
  }

  derived()->call_impl(std::move(target));

  assert((compiler.register_file.allocatable & arg_regs) == 0);
  compiler.register_file.allocatable |= arg_regs;
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
template <typename CBDerived>
void CompilerBase<Adaptor, Derived, Config>::CallBuilderBase<
    CBDerived>::add_ret(ValuePart &vp, CCAssignment cca) noexcept {
  cca.bank = vp.bank();
  cca.size = vp.part_size();
  assigner.assign_ret(cca);
  assert(cca.reg.valid() && "return value must be in register");
  vp.set_value_reg(&compiler, cca.reg);
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
template <typename CBDerived>
void CompilerBase<Adaptor, Derived, Config>::CallBuilderBase<
    CBDerived>::add_ret(ValueRef &vr) noexcept {
  assert(vr.has_assignment());
  u32 part_count = vr.assignment()->part_count;
  for (u32 part_idx = 0; part_idx < part_count; ++part_idx) {
    CCAssignment cca;
    add_ret(vr.part(part_idx), cca);
  }
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
void CompilerBase<Adaptor, Derived, Config>::RetBuilder::add(
    ValuePart &&vp, CCAssignment cca) noexcept {
  cca.bank = vp.bank();
  u32 size = cca.size = vp.part_size();
  assigner.assign_ret(cca);
  assert(cca.reg.valid() && "indirect return value must use sret argument");

  bool needs_ext = cca.int_ext != 0;
  bool ext_sign = cca.int_ext >> 7;
  unsigned ext_bits = cca.int_ext & 0x3f;

  if (vp.is_in_reg(cca.reg)) {
    if (!vp.can_salvage()) {
      compiler.evict_reg(cca.reg);
    } else {
      vp.salvage(&compiler);
    }
    if (needs_ext) {
      compiler.generate_raw_intext(cca.reg, cca.reg, ext_sign, ext_bits, 64);
    }
  } else {
    if (compiler.register_file.is_used(cca.reg)) {
      compiler.evict_reg(cca.reg);
    }
    if (vp.can_salvage()) {
      AsmReg vp_reg = vp.salvage(&compiler);
      if (needs_ext) {
        compiler.generate_raw_intext(cca.reg, vp_reg, ext_sign, ext_bits, 64);
      } else {
        compiler.mov(cca.reg, vp_reg, size);
      }
    } else {
      vp.reload_into_specific_fixed(&compiler, cca.reg);
      if (needs_ext) {
        compiler.generate_raw_intext(cca.reg, cca.reg, ext_sign, ext_bits, 64);
      }
    }
  }
  vp.reset(&compiler);
  assert(!compiler.register_file.is_used(cca.reg));
  compiler.register_file.mark_clobbered(cca.reg);
  compiler.register_file.allocatable &= ~(u64{1} << cca.reg.id());
  ret_regs |= (1ull << cca.reg.id());
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
void CompilerBase<Adaptor, Derived, Config>::RetBuilder::add(
    IRValueRef val) noexcept {
  u32 part_count = compiler.val_parts(val).count();
  ValueRef vr = compiler.val_ref(val);
  for (u32 part_idx = 0; part_idx < part_count; ++part_idx) {
    add(vr.part(part_idx), CCAssignment{});
  }
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
void CompilerBase<Adaptor, Derived, Config>::RetBuilder::ret() noexcept {
  assert((compiler.register_file.allocatable & ret_regs) == 0);
  compiler.register_file.allocatable |= ret_regs;

  compiler.gen_func_epilog();
  compiler.release_regs_after_return();
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
bool CompilerBase<Adaptor, Derived, Config>::compile() {
  // create function symbols
  text_writer.begin_module(assembler);
  text_writer.switch_section(
      assembler.get_section(assembler.get_text_section()));

  assert(func_syms.empty());
  for (const IRFuncRef func : adaptor->funcs()) {
    auto binding = Assembler::SymBinding::GLOBAL;
    if (adaptor->func_has_weak_linkage(func)) {
      binding = Assembler::SymBinding::WEAK;
    } else if (adaptor->func_only_local(func)) {
      binding = Assembler::SymBinding::LOCAL;
    }
    if (adaptor->func_extern(func)) {
      func_syms.push_back(derived()->assembler.sym_add_undef(
          adaptor->func_link_name(func), binding));
    } else {
      func_syms.push_back(derived()->assembler.sym_predef_func(
          adaptor->func_link_name(func), binding));
    }
    derived()->define_func_idx(func, func_syms.size() - 1);
  }

  if (!derived()->hook_post_func_sym_init()) {
    TPDE_LOG_ERR("hook_pust_func_sym_init failed");
    return false;
  }

  // TODO(ts): create function labels?

  bool success = true;

  u32 func_idx = 0;
  for (const IRFuncRef func : adaptor->funcs()) {
    if (adaptor->func_extern(func)) {
      TPDE_LOG_TRACE("Skipping compilation of func {}",
                     adaptor->func_link_name(func));
      ++func_idx;
      continue;
    }

    TPDE_LOG_TRACE("Compiling func {}", adaptor->func_link_name(func));
    if (!derived()->compile_func(func, func_idx)) {
      TPDE_LOG_ERR("Failed to compile function {}",
                   adaptor->func_link_name(func));
      success = false;
    }
    ++func_idx;
  }

  text_writer.flush();
  text_writer.end_module();
  assembler.finalize();

  // TODO(ts): generate object/map?

  return success;
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
void CompilerBase<Adaptor, Derived, Config>::reset() {
  adaptor->reset();

  for (auto &e : stack.fixed_free_lists) {
    e.clear();
  }
  stack.dynamic_free_lists.clear();

  assembler.reset();
  func_syms.clear();
  block_labels.clear();
  personality_syms.clear();
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
void CompilerBase<Adaptor, Derived, Config>::init_assignment(
    IRValueRef value, ValLocalIdx local_idx) noexcept {
  assert(val_assignment(local_idx) == nullptr);
  TPDE_LOG_TRACE("Initializing assignment for value {}",
                 static_cast<u32>(local_idx));

  const auto parts = derived()->val_parts(value);
  const u32 part_count = parts.count();
  assert(part_count > 0);
  auto *assignment = assignments.allocator.allocate(part_count);
  assignments.value_ptrs[static_cast<u32>(local_idx)] = assignment;

  u32 max_part_size = 0;
  for (u32 part_idx = 0; part_idx < part_count; ++part_idx) {
    auto ap = AssignmentPartRef{assignment, part_idx};
    ap.reset();
    ap.set_bank(parts.reg_bank(part_idx));
    const u32 size = parts.size_bytes(part_idx);
    assert(size > 0);
    max_part_size = std::max(max_part_size, size);
    ap.set_part_size(size);
  }

  const auto &liveness = analyzer.liveness_info(local_idx);

  // if there is only one part, try to hand out a fixed assignment
  // if the value is used for longer than one block and there aren't too many
  // definitions in child loops this could interfere with
  // TODO(ts): try out only fixed assignments if the value is live for more
  // than two blocks?
  // TODO(ts): move this to ValuePartRef::alloc_reg to be able to defer this
  // for results?
  if (part_count == 1) {
    const auto &cur_loop =
        analyzer.loop_from_idx(analyzer.block_loop_idx(cur_block_idx));
    auto ap = AssignmentPartRef{assignment, 0};

    auto try_fixed =
        liveness.last > cur_block_idx &&
        cur_loop.definitions_in_childs +
                assignments.cur_fixed_assignment_count[ap.bank().id()] <
            Derived::NUM_FIXED_ASSIGNMENTS[ap.bank().id()];
    if (derived()->try_force_fixed_assignment(value)) {
      try_fixed = assignments.cur_fixed_assignment_count[ap.bank().id()] <
                  Derived::NUM_FIXED_ASSIGNMENTS[ap.bank().id()];
    }

    if (try_fixed) {
      // check if there is a fixed register available
      AsmReg reg = derived()->select_fixed_assignment_reg(ap, value);
      TPDE_LOG_TRACE("Trying to assign fixed reg to value {}",
                     static_cast<u32>(local_idx));

      // TODO: if the register is used, we can free it most of the time, but not
      // always, e.g. for PHI nodes. Detect this case and free_reg otherwise.
      if (!reg.invalid() && !register_file.is_used(reg)) {
        TPDE_LOG_TRACE("Assigning fixed assignment to reg {} for value {}",
                       reg.id(),
                       static_cast<u32>(local_idx));
        ap.set_reg(reg);
        ap.set_register_valid(true);
        ap.set_fixed_assignment(true);
        register_file.mark_used(reg, local_idx, 0);
        register_file.inc_lock_count(reg); // fixed assignments always locked
        register_file.mark_clobbered(reg);
        ++assignments.cur_fixed_assignment_count[ap.bank().id()];
      }
    }
  }

  const auto last_full = liveness.last_full;
  const auto ref_count = liveness.ref_count;

  assert(max_part_size <= 256);
  assignment->max_part_size = max_part_size;
  assignment->pending_free = false;
  assignment->variable_ref = false;
  assignment->stack_variable = false;
  assignment->delay_free = last_full;
  assignment->part_count = part_count;
  assignment->frame_off = 0;
  assignment->references_left = ref_count;
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
void CompilerBase<Adaptor, Derived, Config>::free_assignment(
    ValLocalIdx local_idx, ValueAssignment *assignment) noexcept {
  TPDE_LOG_TRACE("Freeing assignment for value {}",
                 static_cast<u32>(local_idx));

  assert(assignments.value_ptrs[static_cast<u32>(local_idx)] == assignment);
  assignments.value_ptrs[static_cast<u32>(local_idx)] = nullptr;
  const auto is_var_ref = assignment->variable_ref;
  const u32 part_count = assignment->part_count;

  // free registers
  for (u32 part_idx = 0; part_idx < part_count; ++part_idx) {
    auto ap = AssignmentPartRef{assignment, part_idx};
    if (ap.fixed_assignment()) [[unlikely]] {
      const auto reg = ap.get_reg();
      assert(register_file.is_fixed(reg));
      assert(register_file.reg_local_idx(reg) == local_idx);
      assert(register_file.reg_part(reg) == part_idx);
      --assignments.cur_fixed_assignment_count[ap.bank().id()];
      register_file.dec_lock_count_must_zero(reg); // release lock for fixed reg
      register_file.unmark_used(reg);
    } else if (ap.register_valid()) {
      const auto reg = ap.get_reg();
      assert(!register_file.is_fixed(reg));
      register_file.unmark_used(reg);
    }
  }

#ifdef TPDE_ASSERTS
  for (auto reg_id : register_file.used_regs()) {
    assert(register_file.reg_local_idx(AsmReg{reg_id}) != local_idx &&
           "freeing assignment that is still referenced by a register");
  }
#endif

  // variable references do not have a stack slot
  bool has_stack = Config::FRAME_INDEXING_NEGATIVE ? assignment->frame_off < 0
                                                   : assignment->frame_off != 0;
  if (!is_var_ref && has_stack) {
    free_stack_slot(assignment->frame_off, assignment->size());
  }

  assignments.allocator.deallocate(assignment);
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
[[gnu::noinline]] void
    CompilerBase<Adaptor, Derived, Config>::release_assignment(
        ValLocalIdx local_idx, ValueAssignment *assignment) noexcept {
  if (!assignment->delay_free) {
    free_assignment(local_idx, assignment);
    return;
  }

  // need to wait until release
  TPDE_LOG_TRACE("Delay freeing assignment for value {}",
                 static_cast<u32>(local_idx));
  const auto &liveness = analyzer.liveness_info(local_idx);
  auto &free_list_head = assignments.delayed_free_lists[u32(liveness.last)];
  assignment->next_delayed_free_entry = free_list_head;
  assignment->pending_free = true;
  free_list_head = local_idx;
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
void CompilerBase<Adaptor, Derived, Config>::init_variable_ref(
    ValLocalIdx local_idx, u32 var_ref_data) noexcept {
  TPDE_LOG_TRACE("Initializing variable-ref assignment for value {}",
                 static_cast<u32>(local_idx));

  assert(val_assignment(local_idx) == nullptr);
  auto *assignment = assignments.allocator.allocate_slow(1, true);
  assignments.value_ptrs[static_cast<u32>(local_idx)] = assignment;

  assignment->max_part_size = Config::PLATFORM_POINTER_SIZE;
  assignment->variable_ref = true;
  assignment->stack_variable = false;
  assignment->part_count = 1;
  assignment->var_ref_custom_idx = var_ref_data;
  assignment->next_delayed_free_entry = assignments.variable_ref_list;

  assignments.variable_ref_list = local_idx;

  AssignmentPartRef ap{assignment, 0};
  ap.reset();
  ap.set_bank(Config::GP_BANK);
  ap.set_part_size(Config::PLATFORM_POINTER_SIZE);
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
i32 CompilerBase<Adaptor, Derived, Config>::allocate_stack_slot(
    u32 size) noexcept {
  this->stack.frame_used = true;
  unsigned align_bits = 4;
  if (size == 0) {
    return 0; // 0 is the "invalid" stack slot
  } else if (size <= 16) {
    // Align up to next power of two.
    u32 free_list_idx = size == 1 ? 0 : 32 - util::cnt_lz<u32>(size - 1);
    assert(size <= 1u << free_list_idx);
    size = 1 << free_list_idx;
    align_bits = free_list_idx;

    if (!stack.fixed_free_lists[free_list_idx].empty()) {
      auto slot = stack.fixed_free_lists[free_list_idx].back();
      stack.fixed_free_lists[free_list_idx].pop_back();
      return slot;
    }
  } else {
    size = util::align_up(size, 16);
    auto it = stack.dynamic_free_lists.find(size);
    if (it != stack.dynamic_free_lists.end() && !it->second.empty()) {
      const auto slot = it->second.back();
      it->second.pop_back();
      return slot;
    }
  }

  assert(stack.frame_size != ~0u &&
         "cannot allocate stack slot before stack frame is initialized");

  // Align frame_size to align_bits
  for (u32 list_idx = util::cnt_tz(stack.frame_size); list_idx < align_bits;
       list_idx = util::cnt_tz(stack.frame_size)) {
    i32 slot = stack.frame_size;
    if constexpr (Config::FRAME_INDEXING_NEGATIVE) {
      slot = -(slot + (1ull << list_idx));
    }
    stack.fixed_free_lists[list_idx].push_back(slot);
    stack.frame_size += 1ull << list_idx;
  }

  auto slot = stack.frame_size;
  assert(slot != 0 && "stack slot 0 is reserved");
  stack.frame_size += size;

  if constexpr (Config::FRAME_INDEXING_NEGATIVE) {
    slot = -(slot + size);
  }
  return slot;
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
void CompilerBase<Adaptor, Derived, Config>::free_stack_slot(
    u32 slot, u32 size) noexcept {
  if (size == 0) [[unlikely]] {
    assert(slot == 0 && "unexpected slot for zero-sized stack-slot?");
    // Do nothing.
  } else if (size <= 16) [[likely]] {
    u32 free_list_idx = size == 1 ? 0 : 32 - util::cnt_lz<u32>(size - 1);
    stack.fixed_free_lists[free_list_idx].push_back(slot);
  } else {
    size = util::align_up(size, 16);
    stack.dynamic_free_lists[size].push_back(slot);
  }
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
void CompilerBase<Adaptor, Derived, Config>::prologue_assign_arg(
    CCAssigner *cc_assigner,
    u32 arg_idx,
    IRValueRef arg,
    u32 align,
    bool allow_split) noexcept {
  ValueRef vr = derived()->result_ref(arg);
  if (adaptor->cur_arg_is_byval(arg_idx)) {
    CCAssignment cca{
        .byval = true,
        .align = u8(adaptor->cur_arg_byval_align(arg_idx)),
        .size = adaptor->cur_arg_byval_size(arg_idx),
    };
    cc_assigner->assign_arg(cca);
    std::optional<i32> byval_frame_off =
        derived()->prologue_assign_arg_part(vr.part(0), cca);

    if (byval_frame_off) {
      // We need to convert the assignment into a stack variable ref.
      ValLocalIdx local_idx = val_idx(arg);
      // TODO: we shouldn't create the result_ref for such cases in the first
      // place. However, this is not easy to detect up front, it depends on the
      // target and the calling convention whether this is possible.
      vr.reset();
      // Value assignment might have been free'd by ValueRef reset.
      if (ValueAssignment *assignment = val_assignment(local_idx)) {
        free_assignment(local_idx, assignment);
      }
      init_variable_ref(local_idx, 0);
      ValueAssignment *assignment = this->val_assignment(local_idx);
      assignment->stack_variable = true;
      assignment->frame_off = *byval_frame_off;
    }
    return;
  }

  if (adaptor->cur_arg_is_sret(arg_idx)) {
    assert(vr.assignment()->part_count == 1 && "sret must be single-part");
    ValuePartRef vp = vr.part(0);
    CCAssignment cca{
        .sret = true, .bank = vp.bank(), .size = Config::PLATFORM_POINTER_SIZE};
    cc_assigner->assign_arg(cca);
    derived()->prologue_assign_arg_part(std::move(vp), cca);
    return;
  }

  const u32 part_count = vr.assignment()->part_count;
  for (u32 part_idx = 0; part_idx < part_count; ++part_idx) {
    ValuePartRef vp = vr.part(part_idx);
    u32 remaining = part_count < 256 ? part_count - part_idx - 1 : 255;
    CCAssignment cca{
        .consecutive = u8(allow_split ? 0 : remaining),
        .align = u8(part_idx == 0 ? align : 1),
        .bank = vp.bank(),
        .size = vp.part_size(),
    };
    cc_assigner->assign_arg(cca);
    derived()->prologue_assign_arg_part(std::move(vp), cca);
  }
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
typename CompilerBase<Adaptor, Derived, Config>::ValueRef
    CompilerBase<Adaptor, Derived, Config>::val_ref(IRValueRef value) noexcept {
  if (auto special = derived()->val_ref_special(value); special) {
    return ValueRef{this, std::move(*special)};
  }

  const ValLocalIdx local_idx = analyzer.adaptor->val_local_idx(value);
  assert(val_assignment(local_idx) != nullptr && "value use before def");
  return ValueRef{this, local_idx};
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
std::pair<typename CompilerBase<Adaptor, Derived, Config>::ValueRef,
          typename CompilerBase<Adaptor, Derived, Config>::ValuePartRef>
    CompilerBase<Adaptor, Derived, Config>::val_ref_single(
        IRValueRef value) noexcept {
  std::pair<ValueRef, ValuePartRef> res{val_ref(value), this};
  res.second = res.first.part(0);
  return res;
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
typename CompilerBase<Adaptor, Derived, Config>::ValueRef
    CompilerBase<Adaptor, Derived, Config>::result_ref(
        IRValueRef value) noexcept {
  const ValLocalIdx local_idx = analyzer.adaptor->val_local_idx(value);
  if (val_assignment(local_idx) == nullptr) {
    init_assignment(value, local_idx);
  }
  return ValueRef{this, local_idx};
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
std::pair<typename CompilerBase<Adaptor, Derived, Config>::ValueRef,
          typename CompilerBase<Adaptor, Derived, Config>::ValuePartRef>
    CompilerBase<Adaptor, Derived, Config>::result_ref_single(
        IRValueRef value) noexcept {
  std::pair<ValueRef, ValuePartRef> res{result_ref(value), this};
  res.second = res.first.part(0);
  return res;
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
typename CompilerBase<Adaptor, Derived, Config>::ValueRef
    CompilerBase<Adaptor, Derived, Config>::result_ref_alias(
        IRValueRef dst, ValueRef &&src) noexcept {
  const ValLocalIdx local_idx = analyzer.adaptor->val_local_idx(dst);
  assert(!val_assignment(local_idx) && "alias target already defined");
  assert(src.has_assignment() && "alias src must have an assignment");
  // Consider implementing aliases where multiple values share the same
  // assignment, e.g. for implementing trunc as a no-op sharing registers. Live
  // ranges can be merged and reference counts added. However, this needs
  // additional infrastructure to clear one of the value_ptrs.
  assert(src.is_owned() && "alias src must be owned");

  ValueAssignment *assignment = src.assignment();
  u32 part_count = assignment->part_count;
  assert(!assignment->pending_free);
  assert(!assignment->variable_ref);
  assert(!assignment->pending_free);
#ifndef NDEBUG
  {
    const auto &src_liveness = analyzer.liveness_info(src.local_idx());
    assert(!src_liveness.last_full);          // implied by is_owned()
    assert(assignment->references_left == 1); // implied by is_owned()

    // Validate that part configuration is identical.
    const auto parts = derived()->val_parts(dst);
    assert(parts.count() == part_count);
    for (u32 part_idx = 0; part_idx < part_count; ++part_idx) {
      AssignmentPartRef ap{assignment, part_idx};
      assert(parts.reg_bank(part_idx) == ap.bank());
      assert(parts.size_bytes(part_idx) == ap.part_size());
    }
  }
#endif

  // Update local_idx of registers.
  for (u32 part_idx = 0; part_idx < part_count; ++part_idx) {
    AssignmentPartRef ap{assignment, part_idx};
    if (ap.register_valid()) {
      register_file.update_reg_assignment(ap.get_reg(), local_idx, part_idx);
    }
  }

  const auto &liveness = analyzer.liveness_info(local_idx);
  assignment->delay_free = liveness.last_full;
  assignment->references_left = liveness.ref_count;
  assignments.value_ptrs[static_cast<u32>(src.local_idx())] = nullptr;
  assignments.value_ptrs[static_cast<u32>(local_idx)] = assignment;
  src.disown();
  src.reset();

  return ValueRef{this, local_idx};
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
typename CompilerBase<Adaptor, Derived, Config>::ValueRef
    CompilerBase<Adaptor, Derived, Config>::result_ref_stack_slot(
        IRValueRef dst, AssignmentPartRef base, i32 off) noexcept {
  const ValLocalIdx local_idx = analyzer.adaptor->val_local_idx(dst);
  assert(!val_assignment(local_idx) && "new value already defined");
  init_variable_ref(local_idx, 0);
  ValueAssignment *assignment = this->val_assignment(local_idx);
  assignment->stack_variable = true;
  assignment->frame_off = base.variable_stack_off() + off;
  return ValueRef{this, local_idx};
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
void CompilerBase<Adaptor, Derived, Config>::set_value(
    ValuePartRef &val_ref, ScratchReg &scratch) noexcept {
  val_ref.set_value(std::move(scratch));
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
typename CompilerBase<Adaptor, Derived, Config>::AsmReg
    CompilerBase<Adaptor, Derived, Config>::gval_as_reg(
        GenericValuePart &gv) noexcept {
  if (std::holds_alternative<ScratchReg>(gv.state)) {
    return std::get<ScratchReg>(gv.state).cur_reg();
  }
  if (std::holds_alternative<ValuePartRef>(gv.state)) {
    auto &vpr = std::get<ValuePartRef>(gv.state);
    if (vpr.has_reg()) {
      return vpr.cur_reg();
    }
    return vpr.load_to_reg();
  }
  if (auto *expr = std::get_if<typename GenericValuePart::Expr>(&gv.state)) {
    if (expr->has_base() && !expr->has_index() && expr->disp == 0) {
      return expr->base_reg();
    }
    return derived()->gval_expr_as_reg(gv);
  }
  TPDE_UNREACHABLE("gval_as_reg on empty GenericValuePart");
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
typename CompilerBase<Adaptor, Derived, Config>::AsmReg
    CompilerBase<Adaptor, Derived, Config>::gval_as_reg_reuse(
        GenericValuePart &gv, ScratchReg &dst) noexcept {
  AsmReg reg = gval_as_reg(gv);
  if (!dst.has_reg()) {
    if (auto *scratch = std::get_if<ScratchReg>(&gv.state)) {
      dst = std::move(*scratch);
    } else if (auto *val_ref = std::get_if<ValuePartRef>(&gv.state)) {
      if (val_ref->can_salvage()) {
        dst.alloc_specific(val_ref->salvage());
        assert(dst.cur_reg() == reg && "salvaging unsuccessful");
      }
    }
  }
  return reg;
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
Reg CompilerBase<Adaptor, Derived, Config>::select_reg_evict(
    RegBank bank, u64 exclusion_mask) noexcept {
  TPDE_LOG_DBG("select_reg_evict for bank {}", bank.id());
  auto candidates =
      register_file.used & register_file.bank_regs(bank) & ~exclusion_mask;

  Reg candidate = Reg::make_invalid();
  u32 max_score = 0;
  for (auto reg_id : util::BitSetIterator<>(candidates)) {
    Reg reg{reg_id};
    if (register_file.is_fixed(reg)) {
      continue;
    }

    // Must be an evictable value, not a temporary.
    auto local_idx = register_file.reg_local_idx(reg);
    u32 part = register_file.reg_part(Reg{reg});
    assert(local_idx != INVALID_VAL_LOCAL_IDX);
    ValueAssignment *va = val_assignment(local_idx);
    AssignmentPartRef ap{va, part};

    // We want to sort registers by the following (ordered by priority):
    // - stack variable ref (~1 add/sub to reconstruct)
    // - other variable ref (1-2 instrs to reconstruct)
    // - already spilled (no store needed)
    // - last use farthest away (most likely to get spilled anyhow, so there's
    //   not much harm in spilling earlier)
    // - lowest ref-count (least used)
    //
    // TODO: evaluate and refine this heuristic

    // TODO: evict stack variable refs before others
    if (ap.variable_ref()) {
      TPDE_LOG_DBG("  r{} ({}) is variable-ref", reg_id, u32(local_idx));
      candidate = reg;
      break;
    }

    u32 score = 0;
    if (ap.stack_valid()) {
      score |= u32{1} << 31;
    }

    const auto &liveness = analyzer.liveness_info(local_idx);
    u32 last_use_dist = u32(liveness.last) - u32(cur_block_idx);
    score |= (last_use_dist < 0x8000 ? 0x8000 - last_use_dist : 0) << 16;

    u32 refs_left = va->pending_free ? 0 : va->references_left;
    score |= (refs_left < 0xffff ? 0x10000 - refs_left : 1);

    TPDE_LOG_DBG("  r{} ({}:{}) rc={}/{} live={}-{}{} spilled={} score={:#x}",
                 reg_id,
                 u32(local_idx),
                 part,
                 refs_left,
                 liveness.ref_count,
                 u32(liveness.first),
                 u32(liveness.last),
                 &"*"[!liveness.last_full],
                 ap.stack_valid(),
                 score);

    assert(score != 0);
    if (score > max_score) {
      candidate = reg;
      max_score = score;
    }
  }
  if (candidate.invalid()) [[unlikely]] {
    TPDE_FATAL("ran out of registers for scratch registers");
  }
  TPDE_LOG_DBG("  selected r{}", candidate.id());
  evict_reg(candidate);
  return candidate;
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
void CompilerBase<Adaptor, Derived, Config>::reload_to_reg(
    AsmReg dst, AssignmentPartRef ap) noexcept {
  if (!ap.variable_ref()) {
    assert(ap.stack_valid());
    derived()->load_from_stack(dst, ap.frame_off(), ap.part_size());
  } else if (ap.is_stack_variable()) {
    derived()->load_address_of_stack_var(dst, ap);
  } else if constexpr (!Config::DEFAULT_VAR_REF_HANDLING) {
    derived()->load_address_of_var_reference(dst, ap);
  } else {
    TPDE_UNREACHABLE("non-stack-variable needs custom var-ref handling");
  }
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
void CompilerBase<Adaptor, Derived, Config>::allocate_spill_slot(
    AssignmentPartRef ap) noexcept {
  assert(!ap.variable_ref() && "cannot allocate spill slot for variable ref");
  if (ap.assignment()->frame_off == 0) {
    assert(!ap.stack_valid() && "stack-valid set without spill slot");
    ap.assignment()->frame_off = allocate_stack_slot(ap.assignment()->size());
    assert(ap.assignment()->frame_off != 0);
  }
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
void CompilerBase<Adaptor, Derived, Config>::spill(
    AssignmentPartRef ap) noexcept {
  assert(may_change_value_state());
  if (!ap.stack_valid() && !ap.variable_ref()) {
    assert(ap.register_valid() && "cannot spill uninitialized assignment part");
    allocate_spill_slot(ap);
    derived()->spill_reg(ap.get_reg(), ap.frame_off(), ap.part_size());
    ap.set_stack_valid();
  }
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
void CompilerBase<Adaptor, Derived, Config>::evict(
    AssignmentPartRef ap) noexcept {
  assert(may_change_value_state());
  assert(ap.register_valid());
  derived()->spill(ap);
  ap.set_register_valid(false);
  register_file.unmark_used(ap.get_reg());
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
void CompilerBase<Adaptor, Derived, Config>::evict_reg(Reg reg) noexcept {
  assert(may_change_value_state());
  assert(!register_file.is_fixed(reg));
  assert(register_file.reg_local_idx(reg) != INVALID_VAL_LOCAL_IDX);

  ValLocalIdx local_idx = register_file.reg_local_idx(reg);
  auto part = register_file.reg_part(reg);
  AssignmentPartRef evict_part{val_assignment(local_idx), part};
  assert(evict_part.register_valid());
  assert(evict_part.get_reg() == reg);
  derived()->spill(evict_part);
  evict_part.set_register_valid(false);
  register_file.unmark_used(reg);
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
void CompilerBase<Adaptor, Derived, Config>::free_reg(Reg reg) noexcept {
  assert(may_change_value_state());
  assert(!register_file.is_fixed(reg));
  assert(register_file.reg_local_idx(reg) != INVALID_VAL_LOCAL_IDX);

  ValLocalIdx local_idx = register_file.reg_local_idx(reg);
  auto part = register_file.reg_part(reg);
  AssignmentPartRef ap{val_assignment(local_idx), part};
  assert(ap.register_valid());
  assert(ap.get_reg() == reg);
  assert(!ap.modified() || ap.variable_ref());
  ap.set_register_valid(false);
  register_file.unmark_used(reg);
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
typename CompilerBase<Adaptor, Derived, Config>::RegisterFile::RegBitSet
    CompilerBase<Adaptor, Derived, Config>::spill_before_branch(
        bool force_spill) noexcept {
  // since we do not explicitly keep track of register assignments per block,
  // whenever we might branch off to a block that we do not directly compile
  // afterwards (i.e. the register assignments might change in between), we
  // need to spill all registers which are not fixed and remove them from the
  // register state.
  //
  // This leads to worse codegen but saves a significant overhead to
  // store/manage the register assignment for each block (256 bytes/block for
  // x64) and possible compile-time as there might be additional logic to move
  // values around

  // First, we consider the case that the current block only has one successor
  // which is compiled directly after the current one, in which case we do not
  // have to spill anything.
  //
  // Secondly, if the next block has multiple incoming edges, we always have
  // to spill and remove from the register assignment. Otherwise, we
  // only need to spill values if they are alive in any successor which is not
  // the next block.
  //
  // Values which are only read from PHI-Nodes and have no extended lifetimes,
  // do not need to be spilled as they die at the edge.

  using RegBitSet = typename RegisterFile::RegBitSet;

  assert(may_change_value_state());

  const IRBlockRef cur_block_ref = analyzer.block_ref(cur_block_idx);
  // Earliest succeeding block after the current block that is not the
  // immediately succeeding block. Used to determine whether a value needs to
  // be spilled.
  BlockIndex earliest_next_succ = Analyzer<Adaptor>::INVALID_BLOCK_IDX;

  bool must_spill = force_spill;
  if (!must_spill) {
    // We must always spill if no block is immediately succeeding or that block
    // has multiple incoming edges.
    auto next_block_is_succ = false;
    auto next_block_has_multiple_incoming = false;
    u32 succ_count = 0;
    for (const IRBlockRef succ : adaptor->block_succs(cur_block_ref)) {
      ++succ_count;
      BlockIndex succ_idx = analyzer.block_idx(succ);
      if (u32(succ_idx) == u32(cur_block_idx) + 1) {
        next_block_is_succ = true;
        if (analyzer.block_has_multiple_incoming(succ)) {
          next_block_has_multiple_incoming = true;
        }
      } else if (succ_idx > cur_block_idx && succ_idx < earliest_next_succ) {
        earliest_next_succ = succ_idx;
      }
    }

    must_spill = !next_block_is_succ || next_block_has_multiple_incoming;

    if (succ_count == 1 && !must_spill) {
      return RegBitSet{};
    }
  }

  auto release_regs = RegBitSet{};
  // TODO(ts): just use register_file.used_nonfixed_regs()?
  for (auto reg : register_file.used_regs()) {
    auto local_idx = register_file.reg_local_idx(Reg{reg});
    auto part = register_file.reg_part(Reg{reg});
    if (local_idx == INVALID_VAL_LOCAL_IDX) {
      // scratch regs can never be held across blocks
      continue;
    }
    AssignmentPartRef ap{val_assignment(local_idx), part};
    if (ap.fixed_assignment()) {
      // fixed registers do not need to be spilled
      continue;
    }

    // Remove from register assignment if the next block cannot rely on the
    // value being in the specific register.
    if (must_spill) {
      release_regs |= RegBitSet{1ull} << reg;
    }

    if (!ap.modified() || ap.variable_ref()) {
      // No need to spill values that were already spilled or are variable refs.
      continue;
    }

    const auto &liveness = analyzer.liveness_info(local_idx);
    if (liveness.last <= cur_block_idx) {
      // No need to spill value if it dies immediately after the block.
      continue;
    }

    // If the value is live in a different block, we might need to spill it.
    // Cases:
    //  - Live in successor that precedes current block => ignore, value must
    //    have been spilled already due to RPO.
    //  - Live only in successor that immediately follows this block => no need
    //    to spill, value can be kept in register, register state for next block
    //    is still valid.
    //  - Live in successor in some distance after current block => spill
    if (must_spill || earliest_next_succ <= liveness.last) {
      spill(ap);
    }
  }

  return release_regs;
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
void CompilerBase<Adaptor, Derived, Config>::release_spilled_regs(
    typename RegisterFile::RegBitSet regs) noexcept {
  assert(may_change_value_state());

  // TODO(ts): needs changes for other RegisterFile impls
  for (auto reg_id : util::BitSetIterator<>{regs & register_file.used}) {
    if (!register_file.is_fixed(Reg{reg_id})) {
      free_reg(Reg{reg_id});
    }
  }
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
void CompilerBase<Adaptor, Derived, Config>::
    release_regs_after_return() noexcept {
  // we essentially have to free all non-fixed registers
  for (auto reg_id : register_file.used_regs()) {
    if (!register_file.is_fixed(Reg{reg_id})) {
      free_reg(Reg{reg_id});
    }
  }
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
template <typename Jump>
void CompilerBase<Adaptor, Derived, Config>::generate_branch_to_block(
    Jump jmp, IRBlockRef target, bool needs_split, bool last_inst) noexcept {
  BlockIndex target_idx = this->analyzer.block_idx(target);
  Label target_label = this->block_labels[u32(target_idx)];
  if (!needs_split) {
    move_to_phi_nodes(target_idx);
    if (!last_inst || target_idx != this->next_block()) {
      derived()->generate_raw_jump(jmp, target_label);
    }
  } else {
    Label tmp_label = this->text_writer.label_create();
    derived()->generate_raw_jump(derived()->invert_jump(jmp), tmp_label);
    move_to_phi_nodes(target_idx);
    derived()->generate_raw_jump(Jump::jmp, target_label);
    this->label_place(tmp_label);
  }
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
void CompilerBase<Adaptor, Derived, Config>::generate_uncond_branch(
    IRBlockRef target) noexcept {
  auto spilled = spill_before_branch();
  begin_branch_region();

  generate_branch_to_block(Derived::Jump::jmp, target, false, true);

  end_branch_region();
  release_spilled_regs(spilled);
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
template <typename Jump>
void CompilerBase<Adaptor, Derived, Config>::generate_cond_branch(
    Jump jmp, IRBlockRef true_target, IRBlockRef false_target) noexcept {
  IRBlockRef next = analyzer.block_ref(next_block());

  bool true_needs_split = branch_needs_split(true_target);
  bool false_needs_split = branch_needs_split(false_target);

  auto spilled = spill_before_branch();
  begin_branch_region();

  if (next == true_target || (next != false_target && true_needs_split)) {
    generate_branch_to_block(
        derived()->invert_jump(jmp), false_target, false_needs_split, false);
    generate_branch_to_block(Derived::Jump::jmp, true_target, false, true);
  } else if (next == false_target) {
    generate_branch_to_block(jmp, true_target, true_needs_split, false);
    generate_branch_to_block(Derived::Jump::jmp, false_target, false, true);
  } else {
    assert(!true_needs_split);
    generate_branch_to_block(jmp, true_target, false, false);
    generate_branch_to_block(Derived::Jump::jmp, false_target, false, true);
  }

  end_branch_region();
  release_spilled_regs(spilled);
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
void CompilerBase<Adaptor, Derived, Config>::generate_switch(
    ScratchReg &&cond,
    u32 width,
    IRBlockRef default_block,
    std::span<const std::pair<u64, IRBlockRef>> cases) noexcept {
  // This function takes cond as a ScratchReg as opposed to a ValuePart, because
  // the ValueRef for the condition must be ref-counted before we enter the
  // branch region.

  assert(width <= 64);
  // We don't support sections with more than 4 GiB, so switches with more than
  // 4G cases are impossible to support.
  assert(cases.size() < UINT32_MAX && "large switches are unsupported");

  AsmReg cmp_reg = cond.cur_reg();
  bool width_is_32 = width <= 32;
  if (u32 dst_width = util::align_up(width, 32); width != dst_width) {
    derived()->generate_raw_intext(cmp_reg, cmp_reg, false, width, dst_width);
  }

  // We must not evict any registers in the branching code, as we don't track
  // the individual value states per block. Hence, we must not allocate any
  // registers (e.g., for constants, jump table address) below.
  ScratchReg tmp_scratch{this};
  AsmReg tmp_reg = tmp_scratch.alloc_gp();

  const auto spilled = this->spill_before_branch();
  this->begin_branch_region();

  tpde::util::SmallVector<tpde::Label, 64> case_labels;
  // Labels that need an intermediate block to setup registers. This is
  // separate, because most switch targets don't need this.
  tpde::util::SmallVector<std::pair<tpde::Label, IRBlockRef>, 64> case_blocks;
  for (auto i = 0u; i < cases.size(); ++i) {
    // If the target might need additional register moves, we can't branch there
    // immediately.
    // TODO: more precise condition?
    BlockIndex target = this->analyzer.block_idx(cases[i].second);
    if (analyzer.block_has_phis(target)) {
      case_labels.push_back(this->text_writer.label_create());
      case_blocks.emplace_back(case_labels.back(), cases[i].second);
    } else {
      case_labels.push_back(this->block_labels[u32(target)]);
    }
  }

  const auto default_label = this->text_writer.label_create();

  const auto build_range = [&,
                            this](size_t begin, size_t end, const auto &self) {
    assert(begin <= end);
    const auto num_cases = end - begin;
    if (num_cases <= 4) {
      // if there are four or less cases we just compare the values
      // against each of them
      for (auto i = 0u; i < num_cases; ++i) {
        derived()->switch_emit_cmpeq(case_labels[begin + i],
                                     cmp_reg,
                                     tmp_reg,
                                     cases[begin + i].first,
                                     width_is_32);
      }

      derived()->generate_raw_jump(Derived::Jump::jmp, default_label);
      return;
    }

    // check if the density of the values is high enough to warrant building
    // a jump table
    auto range = cases[end - 1].first - cases[begin].first;
    // we will get wrong results if range is -1 so skip the jump table if
    // that is the case
    if (range != 0xFFFF'FFFF'FFFF'FFFF && (range / num_cases) < 8) {
      // for gcc, it seems that if there are less than 8 values per
      // case it will build a jump table so we do that, too

      // the actual range is one greater than the result we get from
      // subtracting so adjust for that
      range += 1;

      tpde::util::SmallVector<tpde::Label, 32> label_vec;
      std::span<tpde::Label> labels;
      if (range == num_cases) {
        labels = std::span{case_labels.begin() + begin, num_cases};
      } else {
        label_vec.resize(range, default_label);
        for (auto i = 0u; i < num_cases; ++i) {
          label_vec[cases[begin + i].first - cases[begin].first] =
              case_labels[begin + i];
        }
        labels = std::span{label_vec.begin(), range};
      }

      // Give target the option to emit a jump table.
      if (derived()->switch_emit_jump_table(default_label,
                                            labels,
                                            cmp_reg,
                                            tmp_reg,
                                            cases[begin].first,
                                            cases[end - 1].first,
                                            width_is_32)) {
        return;
      }
    }

    // do a binary search step
    const auto half_len = num_cases / 2;
    const auto half_value = cases[begin + half_len].first;
    const auto gt_label = this->text_writer.label_create();

    // this will cmp against the input value, jump to the case if it is
    // equal or to gt_label if the value is greater. Otherwise it will
    // fall-through
    derived()->switch_emit_binary_step(case_labels[begin + half_len],
                                       gt_label,
                                       cmp_reg,
                                       tmp_reg,
                                       half_value,
                                       width_is_32);
    // search the lower half
    self(begin, begin + half_len, self);

    // and the upper half
    this->label_place(gt_label);
    self(begin + half_len + 1, end, self);
  };

  build_range(0, case_labels.size(), build_range);

  // write out the labels
  this->label_place(default_label);
  derived()->generate_branch_to_block(
      Derived::Jump::jmp, default_block, false, false);

  for (const auto &[label, target] : case_blocks) {
    // Branch predictors typically have problems if too many branches follow too
    // closely. Ensure a minimum alignment.
    this->text_writer.align(8);
    this->label_place(label);
    derived()->generate_branch_to_block(
        Derived::Jump::jmp, target, false, false);
  }

  this->end_branch_region();
  this->release_spilled_regs(spilled);
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
void CompilerBase<Adaptor, Derived, Config>::move_to_phi_nodes_impl(
    BlockIndex target) noexcept {
  // PHI-nodes are always moved to their stack-slot (unless they are fixed)
  //
  // However, we need to take care of PHI-dependencies (cycles and chains)
  // as to not overwrite values which might be needed.
  //
  // In most cases, we expect the number of PHIs to be small but we want to
  // stay reasonably efficient even with larger numbers of PHIs

  struct ScratchWrapper {
    Derived *self;
    AsmReg cur_reg = AsmReg::make_invalid();
    bool backed_up = false;
    bool was_modified = false;
    u8 part = 0;
    ValLocalIdx local_idx = INVALID_VAL_LOCAL_IDX;

    ScratchWrapper(Derived *self) : self{self} {}

    ~ScratchWrapper() { reset(); }

    void reset() {
      if (cur_reg.invalid()) {
        return;
      }

      self->register_file.unmark_fixed(cur_reg);
      self->register_file.unmark_used(cur_reg);

      if (backed_up) {
        // restore the register state
        // TODO(ts): do we actually need the reload?
        auto *assignment = self->val_assignment(local_idx);
        // check if the value was free'd, then we dont need to restore
        // it
        if (assignment) {
          auto ap = AssignmentPartRef{assignment, part};
          if (!ap.variable_ref()) {
            // TODO(ts): assert that this always happens?
            assert(ap.stack_valid());
            self->load_from_stack(cur_reg, ap.frame_off(), ap.part_size());
          }
          ap.set_reg(cur_reg);
          ap.set_register_valid(true);
          ap.set_modified(was_modified);
          self->register_file.mark_used(cur_reg, local_idx, part);
        }
        backed_up = false;
      }
      cur_reg = AsmReg::make_invalid();
    }

    AsmReg alloc_from_bank(RegBank bank) {
      if (cur_reg.valid() && self->register_file.reg_bank(cur_reg) == bank) {
        return cur_reg;
      }
      if (cur_reg.valid()) {
        reset();
      }

      // TODO(ts): try to first find a non callee-saved/clobbered
      // register...
      auto &reg_file = self->register_file;
      auto reg = reg_file.find_first_free_excluding(bank, 0);
      if (reg.invalid()) {
        // TODO(ts): use clock here?
        reg = reg_file.find_first_nonfixed_excluding(bank, 0);
        if (reg.invalid()) {
          TPDE_FATAL("ran out of registers for scratch registers");
        }

        backed_up = true;
        local_idx = reg_file.reg_local_idx(reg);
        part = reg_file.reg_part(reg);
        AssignmentPartRef ap{self->val_assignment(local_idx), part};
        was_modified = ap.modified();
        // TODO(ts): this does not spill for variable refs
        // We don't use evict_reg here, as we know that we can't change the
        // value state.
        assert(ap.register_valid() && ap.get_reg() == reg);
        if (!ap.stack_valid() && !ap.variable_ref()) {
          self->allocate_spill_slot(ap);
          self->spill_reg(ap.get_reg(), ap.frame_off(), ap.part_size());
          ap.set_stack_valid();
        }
        ap.set_register_valid(false);
        reg_file.unmark_used(reg);
      }

      reg_file.mark_used(reg, INVALID_VAL_LOCAL_IDX, 0);
      reg_file.mark_clobbered(reg);
      reg_file.mark_fixed(reg);
      cur_reg = reg;
      return reg;
    }

    ScratchWrapper &operator=(const ScratchWrapper &) = delete;
    ScratchWrapper &operator=(ScratchWrapper &&) = delete;
  };

  IRBlockRef target_ref = analyzer.block_ref(target);
  IRBlockRef cur_ref = analyzer.block_ref(cur_block_idx);

  // collect all the nodes
  struct NodeEntry {
    IRValueRef phi;
    IRValueRef incoming_val;
    ValLocalIdx phi_local_idx;
    // local idx of same-block phi node that needs special handling
    ValLocalIdx incoming_phi_local_idx = INVALID_VAL_LOCAL_IDX;
    // bool incoming_is_phi;
    u32 ref_count;

    bool operator<(const NodeEntry &other) const noexcept {
      return phi_local_idx < other.phi_local_idx;
    }

    bool operator<(ValLocalIdx other) const noexcept {
      return phi_local_idx < other;
    }
  };

  util::SmallVector<NodeEntry, 16> nodes;
  for (IRValueRef phi : adaptor->block_phis(target_ref)) {
    ValLocalIdx phi_local_idx = adaptor->val_local_idx(phi);
    auto incoming = adaptor->val_as_phi(phi).incoming_val_for_block(cur_ref);
    nodes.emplace_back(NodeEntry{
        .phi = phi, .incoming_val = incoming, .phi_local_idx = phi_local_idx});
  }

  // We check that the block has phi nodes before getting here.
  assert(!nodes.empty() && "block marked has having phi nodes has none");

  ScratchWrapper scratch{derived()};
  const auto move_to_phi = [this, &scratch](IRValueRef phi,
                                            IRValueRef incoming_val) {
    auto phi_vr = derived()->result_ref(phi);
    auto val_vr = derived()->val_ref(incoming_val);
    if (phi == incoming_val) {
      return;
    }

    u32 part_count = phi_vr.assignment()->part_count;
    for (u32 i = 0; i < part_count; ++i) {
      AssignmentPartRef phi_ap{phi_vr.assignment(), i};
      ValuePartRef val_vpr = val_vr.part(i);

      if (phi_ap.fixed_assignment()) {
        if (AsmReg reg = val_vpr.cur_reg_unlocked(); reg.valid()) {
          derived()->mov(phi_ap.get_reg(), reg, phi_ap.part_size());
        } else {
          val_vpr.reload_into_specific_fixed(phi_ap.get_reg());
        }
      } else {
        AsmReg reg = val_vpr.cur_reg_unlocked();
        if (!reg.valid()) {
          reg = scratch.alloc_from_bank(val_vpr.bank());
          val_vpr.reload_into_specific_fixed(reg);
        }
        allocate_spill_slot(phi_ap);
        derived()->spill_reg(reg, phi_ap.frame_off(), phi_ap.part_size());
        phi_ap.set_stack_valid();
      }
    }
  };

  if (nodes.size() == 1) {
    move_to_phi(nodes[0].phi, nodes[0].incoming_val);
    return;
  }

  // sort so we can binary search later
  std::sort(nodes.begin(), nodes.end());

  // fill in the refcount
  auto all_zero_ref = true;
  for (auto &node : nodes) {
    // We don't need to do anything for PHIs that don't reference other PHIs or
    // self-referencing PHIs.
    bool incoming_is_phi = adaptor->val_is_phi(node.incoming_val);
    if (!incoming_is_phi || node.incoming_val == node.phi) {
      continue;
    }

    ValLocalIdx inc_local_idx = adaptor->val_local_idx(node.incoming_val);
    auto it = std::lower_bound(nodes.begin(), nodes.end(), inc_local_idx);
    if (it == nodes.end() || it->phi != node.incoming_val) {
      // Incoming value is a PHI node, but it's not from our block, so we don't
      // need to be particularly careful when assigning values.
      continue;
    }
    node.incoming_phi_local_idx = inc_local_idx;
    ++it->ref_count;
    all_zero_ref = false;
  }

  if (all_zero_ref) {
    // no cycles/chain that we need to take care of
    for (auto &node : nodes) {
      move_to_phi(node.phi, node.incoming_val);
    }
    return;
  }

  // TODO(ts): this is rather inefficient...
  util::SmallVector<u32, 32> ready_indices;
  ready_indices.reserve(nodes.size());
  util::SmallBitSet<256> waiting_nodes;
  waiting_nodes.resize(nodes.size());
  for (u32 i = 0; i < nodes.size(); ++i) {
    if (nodes[i].ref_count) {
      waiting_nodes.mark_set(i);
    } else {
      ready_indices.push_back(i);
    }
  }

  u32 handled_count = 0;
  u32 cur_tmp_part_count = 0;
  i32 cur_tmp_slot = 0;
  u32 cur_tmp_slot_size = 0;
  IRValueRef cur_tmp_val = Adaptor::INVALID_VALUE_REF;
  ScratchWrapper tmp_reg1{derived()}, tmp_reg2{derived()};

  const auto move_from_tmp_phi = [&](IRValueRef target_phi) {
    auto phi_vr = val_ref(target_phi);
    if (cur_tmp_part_count <= 2) {
      AssignmentPartRef ap{phi_vr.assignment(), 0};
      assert(!tmp_reg1.cur_reg.invalid());
      if (ap.fixed_assignment()) {
        derived()->mov(ap.get_reg(), tmp_reg1.cur_reg, ap.part_size());
      } else {
        derived()->spill_reg(tmp_reg1.cur_reg, ap.frame_off(), ap.part_size());
      }

      if (cur_tmp_part_count == 2) {
        AssignmentPartRef ap_high{phi_vr.assignment(), 1};
        assert(!ap_high.fixed_assignment());
        assert(!tmp_reg2.cur_reg.invalid());
        derived()->spill_reg(
            tmp_reg2.cur_reg, ap_high.frame_off(), ap_high.part_size());
      }
      return;
    }

    for (u32 i = 0; i < cur_tmp_part_count; ++i) {
      AssignmentPartRef phi_ap{phi_vr.assignment(), i};
      assert(!phi_ap.fixed_assignment());

      auto slot_off = cur_tmp_slot + phi_ap.part_off();
      auto reg = tmp_reg1.alloc_from_bank(phi_ap.bank());
      derived()->load_from_stack(reg, slot_off, phi_ap.part_size());
      derived()->spill_reg(reg, phi_ap.frame_off(), phi_ap.part_size());
    }
  };

  while (handled_count != nodes.size()) {
    if (ready_indices.empty()) {
      // need to break a cycle
      auto cur_idx_opt = waiting_nodes.first_set();
      assert(cur_idx_opt);
      auto cur_idx = *cur_idx_opt;
      assert(nodes[cur_idx].ref_count == 1);
      assert(cur_tmp_val == Adaptor::INVALID_VALUE_REF);

      auto phi_val = nodes[cur_idx].phi;
      auto phi_vr = this->val_ref(phi_val);
      auto *assignment = phi_vr.assignment();
      cur_tmp_part_count = assignment->part_count;
      cur_tmp_val = phi_val;

      if (cur_tmp_part_count > 2) {
        // use a stack slot to store the temporaries
        cur_tmp_slot_size = assignment->size();
        cur_tmp_slot = allocate_stack_slot(cur_tmp_slot_size);

        for (u32 i = 0; i < cur_tmp_part_count; ++i) {
          auto ap = AssignmentPartRef{assignment, i};
          assert(!ap.fixed_assignment());
          auto slot_off = cur_tmp_slot + ap.part_off();

          if (ap.register_valid()) {
            auto reg = ap.get_reg();
            derived()->spill_reg(reg, slot_off, ap.part_size());
          } else {
            auto reg = tmp_reg1.alloc_from_bank(ap.bank());
            assert(ap.stack_valid());
            derived()->load_from_stack(reg, ap.frame_off(), ap.part_size());
            derived()->spill_reg(reg, slot_off, ap.part_size());
          }
        }
      } else {
        // TODO(ts): if the PHI is not fixed, then we can just reuse its
        // register if it has one
        auto phi_vpr = phi_vr.part(0);
        auto reg = tmp_reg1.alloc_from_bank(phi_vpr.bank());
        phi_vpr.reload_into_specific_fixed(this, reg);

        if (cur_tmp_part_count == 2) {
          // TODO(ts): just change the part ref on the lower ref?
          auto phi_vpr_high = phi_vr.part(1);
          auto reg_high = tmp_reg2.alloc_from_bank(phi_vpr_high.bank());
          phi_vpr_high.reload_into_specific_fixed(this, reg_high);
        }
      }

      nodes[cur_idx].ref_count = 0;
      ready_indices.push_back(cur_idx);
      waiting_nodes.mark_unset(cur_idx);
    }

    for (u32 i = 0; i < ready_indices.size(); ++i) {
      ++handled_count;
      auto cur_idx = ready_indices[i];
      auto phi_val = nodes[cur_idx].phi;
      IRValueRef incoming_val = nodes[cur_idx].incoming_val;
      if (incoming_val == phi_val) {
        // no need to do anything, except ref-counting
        (void)val_ref(incoming_val);
        (void)val_ref(phi_val);
        continue;
      }

      if (incoming_val == cur_tmp_val) {
        move_from_tmp_phi(phi_val);

        if (cur_tmp_part_count > 2) {
          free_stack_slot(cur_tmp_slot, cur_tmp_slot_size);
          cur_tmp_slot = 0xFFFF'FFFF;
          cur_tmp_slot_size = 0;
        }
        cur_tmp_val = Adaptor::INVALID_VALUE_REF;
        // skip the code below as the ref count of the temp val is
        // already 0 and we don't want to reinsert it into the ready
        // list
        continue;
      }

      move_to_phi(phi_val, incoming_val);

      if (nodes[cur_idx].incoming_phi_local_idx == INVALID_VAL_LOCAL_IDX) {
        continue;
      }

      auto it = std::lower_bound(
          nodes.begin(), nodes.end(), nodes[cur_idx].incoming_phi_local_idx);
      assert(it != nodes.end() && it->phi == incoming_val &&
             "incoming_phi_local_idx set incorrectly");

      assert(it->ref_count > 0);
      if (--it->ref_count == 0) {
        auto node_idx = static_cast<u32>(it - nodes.begin());
        ready_indices.push_back(node_idx);
        waiting_nodes.mark_unset(node_idx);
      }
    }
    ready_indices.clear();
  }
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
typename CompilerBase<Adaptor, Derived, Config>::BlockIndex
    CompilerBase<Adaptor, Derived, Config>::next_block() const noexcept {
  return static_cast<BlockIndex>(static_cast<u32>(cur_block_idx) + 1);
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
SymRef CompilerBase<Adaptor, Derived, Config>::get_personality_sym() noexcept {
  SymRef personality_sym;
  if (this->adaptor->cur_needs_unwind_info()) {
    SymRef personality_func = derived()->cur_personality_func();
    if (personality_func.valid()) {
      for (const auto &[fn_sym, ptr_sym] : personality_syms) {
        if (fn_sym == personality_func) {
          personality_sym = ptr_sym;
          break;
        }
      }

      if (!personality_sym.valid()) {
        // create symbol that contains the address of the personality
        // function
        u32 off;
        static constexpr std::array<u8, 8> zero{};

        auto rodata = this->assembler.get_data_section(true, true);
        personality_sym = this->assembler.sym_def_data(
            rodata, "", zero, 8, Assembler::SymBinding::LOCAL, &off);
        this->assembler.reloc_abs(rodata, personality_func, off, 0);

        personality_syms.emplace_back(personality_func, personality_sym);
      }
    }
  }
  return personality_sym;
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
bool CompilerBase<Adaptor, Derived, Config>::compile_func(
    const IRFuncRef func, const u32 func_idx) noexcept {
  if (!adaptor->switch_func(func)) {
    return false;
  }
  derived()->analysis_start();
  analyzer.switch_func(func);
  derived()->analysis_end();

#ifndef NDEBUG
  stack.frame_size = ~0u;
#endif
  for (auto &e : stack.fixed_free_lists) {
    e.clear();
  }
  stack.dynamic_free_lists.clear();
  // TODO: sort out the inconsistency about adaptor vs. compiler methods.
  stack.has_dynamic_alloca = this->adaptor->cur_has_dynamic_alloca();
  stack.is_leaf_function = !derived()->cur_func_may_emit_calls();
  stack.generated_call = false;
  stack.frame_used = false;

  assignments.cur_fixed_assignment_count = {};
  assert(std::ranges::none_of(assignments.value_ptrs, std::identity{}));
  if (assignments.value_ptrs.size() < analyzer.liveness.size()) {
    assignments.value_ptrs.resize(analyzer.liveness.size());
  }

  assignments.allocator.reset();
  assignments.variable_ref_list = INVALID_VAL_LOCAL_IDX;
  assignments.delayed_free_lists.clear();
  assignments.delayed_free_lists.resize(analyzer.block_layout.size(),
                                        INVALID_VAL_LOCAL_IDX);

  cur_block_idx =
      static_cast<BlockIndex>(analyzer.block_idx(adaptor->cur_entry_block()));

  register_file.reset();
#ifndef NDEBUG
  generating_branch = false;
#endif

  // Simple heuristic for initial allocation size
  u32 expected_code_size = 0x8 * analyzer.num_insts + 0x40;
  this->text_writer.begin_func(/*alignment=*/16, expected_code_size);

  derived()->start_func(func_idx);

  block_labels.clear();
  block_labels.resize_uninitialized(analyzer.block_layout.size());
  for (u32 i = 0; i < analyzer.block_layout.size(); ++i) {
    block_labels[i] = text_writer.label_create();
  }

  // TODO(ts): place function label
  // TODO(ts): make function labels optional?

  CCAssigner *cc_assigner = derived()->cur_cc_assigner();
  assert(cc_assigner != nullptr);

  register_file.allocatable = cc_assigner->get_ccinfo().allocatable_regs;

  cc_assigner->reset();
  // Temporarily prevent argument registers from being assigned.
  const CCInfo &cc_info = cc_assigner->get_ccinfo();
  assert((cc_info.allocatable_regs & cc_info.arg_regs) == cc_info.arg_regs &&
         "argument registers must also be allocatable");
  this->register_file.allocatable &= ~cc_info.arg_regs;

  // Begin prologue, prepare for handling arguments.
  derived()->prologue_begin(cc_assigner);
  u32 arg_idx = 0;
  for (const IRValueRef arg : this->adaptor->cur_args()) {
    // Init assignment for all arguments. This can be substituted for more
    // complex mappings of arguments to value parts.
    derived()->prologue_assign_arg(cc_assigner, arg_idx++, arg);
  }
  // Finish prologue, storing relevant data from the argument cc_assigner.
  derived()->prologue_end(cc_assigner);

  this->register_file.allocatable |= cc_info.arg_regs;

  for (const IRValueRef alloca : adaptor->cur_static_allocas()) {
    auto size = adaptor->val_alloca_size(alloca);
    size = util::align_up(size, adaptor->val_alloca_align(alloca));

    ValLocalIdx local_idx = adaptor->val_local_idx(alloca);
    init_variable_ref(local_idx, 0);
    ValueAssignment *assignment = val_assignment(local_idx);
    assignment->stack_variable = true;
    assignment->frame_off = allocate_stack_slot(size);
  }

  if constexpr (!Config::DEFAULT_VAR_REF_HANDLING) {
    derived()->setup_var_ref_assignments();
  }

  for (u32 i = 0; i < analyzer.block_layout.size(); ++i) {
    const auto block_ref = analyzer.block_layout[i];
    TPDE_LOG_TRACE(
        "Compiling block {} ({})", i, adaptor->block_fmt_ref(block_ref));
    if (!derived()->compile_block(block_ref, i)) [[unlikely]] {
      TPDE_LOG_ERR("Failed to compile block {} ({})",
                   i,
                   adaptor->block_fmt_ref(block_ref));
      // Ensure invariant that value_ptrs only contains nullptr at the end.
      assignments.value_ptrs.clear();
      return false;
    }
  }

  // Reset all variable-ref assignment pointers to nullptr.
  ValLocalIdx variable_ref_list = assignments.variable_ref_list;
  while (variable_ref_list != INVALID_VAL_LOCAL_IDX) {
    u32 idx = u32(variable_ref_list);
    ValLocalIdx next = assignments.value_ptrs[idx]->next_delayed_free_entry;
    assignments.value_ptrs[idx] = nullptr;
    variable_ref_list = next;
  }

  assert(std::ranges::none_of(assignments.value_ptrs, std::identity{}) &&
         "found non-freed ValueAssignment, maybe missing ref-count?");

  derived()->finish_func(func_idx);
  this->text_writer.finish_func();

  return true;
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
bool CompilerBase<Adaptor, Derived, Config>::compile_block(
    const IRBlockRef block, const u32 block_idx) noexcept {
  cur_block_idx =
      static_cast<typename Analyzer<Adaptor>::BlockIndex>(block_idx);

  label_place(block_labels[block_idx]);
  auto &&val_range = adaptor->block_insts(block);
  auto end = val_range.end();
  for (auto it = val_range.begin(); it != end; ++it) {
    const IRInstRef inst = *it;
    if (this->adaptor->inst_fused(inst)) {
      continue;
    }

    auto it_cpy = it;
    ++it_cpy;
    if (!derived()->compile_inst(inst, InstRange{.from = it_cpy, .to = end}))
        [[unlikely]] {
      TPDE_LOG_ERR("Failed to compile instruction {}",
                   this->adaptor->inst_fmt_ref(inst));
      return false;
    }
  }

#ifndef NDEBUG
  // Some consistency checks. Register assignment information must match, all
  // used registers must have an assignment (no temporaries across blocks), and
  // fixed registers must be fixed assignments.
  for (auto reg_id : register_file.used_regs()) {
    Reg reg{reg_id};
    assert(register_file.reg_local_idx(reg) != INVALID_VAL_LOCAL_IDX);
    AssignmentPartRef ap{val_assignment(register_file.reg_local_idx(reg)),
                         register_file.reg_part(reg)};
    assert(ap.register_valid());
    assert(ap.get_reg() == reg);
    assert(!register_file.is_fixed(reg) || ap.fixed_assignment());
  }
#endif

  if (static_cast<u32>(assignments.delayed_free_lists[block_idx]) != ~0u) {
    auto list_entry = assignments.delayed_free_lists[block_idx];
    while (static_cast<u32>(list_entry) != ~0u) {
      auto *assignment = assignments.value_ptrs[static_cast<u32>(list_entry)];
      auto next_entry = assignment->next_delayed_free_entry;
      derived()->free_assignment(list_entry, assignment);
      list_entry = next_entry;
    }
  }
  return true;
}

} // namespace tpde
