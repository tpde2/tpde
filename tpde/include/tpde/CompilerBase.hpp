// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include <algorithm>
#include <functional>
#include <unordered_map>

#include "Analyzer.hpp"
#include "Compiler.hpp"
#include "CompilerConfig.hpp"
#include "IRAdaptor.hpp"
#include "tpde/AssignmentPartRef.hpp"
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
  Reg reg = Reg::make_invalid();

  /// If non-zero, indicates that this and the next N values must be
  /// assigned to consecutive registers, or to the stack. The following
  /// values must be in the same register bank as this value. u8 is sufficient,
  /// no architecture has more than 255 parameter registers in a single bank.
  u8 consecutive = 0;
  bool sret : 1 = false; ///< Argument is return value pointer.
  bool sext : 1 = false; ///< Sign-extend single integer.
  bool zext : 1 = false; ///< Zero-extend single integer.
  bool byval : 1 = false;
  u8 align = 0;
  u8 size = 0;
  RegBank bank = RegBank{};
  u32 byval_align = 0;
  u32 byval_size = 0; // only used if byval is set
  u32 stack_off = 0;  // only used if reg is invalid
};

struct CCInfo {
  // TODO: use RegBitSet
  const u64 allocatable_regs;
  const u64 callee_saved_regs;
  /// Possible argument registers; these registers will not be allocated until
  /// all arguments have been assigned.
  const u64 arg_regs;
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

  using Assembler = typename Config::Assembler;
  using AsmReg = typename Config::AsmReg;

  using RegisterFile = RegisterFile<Config::NUM_BANKS, 32>;

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
  Assembler assembler;
  Assembler::SectionWriter text_writer;
  // TODO(ts): smallvector?
  std::vector<typename Assembler::SymRef> func_syms;
  // TODO(ts): combine this with the block vectors in the analyzer to save on
  // allocations
  util::SmallVector<typename Assembler::Label> block_labels;

  util::SmallVector<
      std::pair<typename Assembler::SymRef, typename Assembler::SymRef>,
      4>
      personality_syms = {};

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
      byval
    };

    explicit CallArg(IRValueRef value,
                     Flag flags = Flag::none,
                     u32 byval_align = 0,
                     u32 byval_size = 0)
        : value(value),
          flag(flags),
          byval_align(byval_align),
          byval_size(byval_size) {}

    IRValueRef value;
    Flag flag;
    u32 byval_align;
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
    // void call_impl(std::variant<Assembler::SymRef, ValuePart> &&) noexcept;
    CBDerived *derived() noexcept { return static_cast<CBDerived *>(this); }

    void add_arg(ValuePart &&vp, CCAssignment cca) noexcept;
    void add_arg(CallArg &&arg) noexcept;

    // evict registers, do call, reset stack frame
    void call(std::variant<typename Assembler::SymRef, ValuePart>) noexcept;

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
    void add(IRValueRef val, CallArg::Flag flag = CallArg::Flag::none) noexcept;

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
  ///   in between the calls.
  ///
  /// \returns Whether the compilation was successful
  bool compile();

  /// Reset any leftover data from the previous compilation such that it will
  /// not affect the next compilation
  void reset();

  /// Get CCAssigner for current function.
  CCAssigner *cur_cc_assigner() noexcept { return &default_cc_assigner; }

  void init_assignment(IRValueRef value, ValLocalIdx local_idx) noexcept;

  /// Frees an assignment, its stack slot and registers
  void free_assignment(ValLocalIdx local_idx) noexcept;

  /// Init a variable-ref assignment
  void init_variable_ref(ValLocalIdx local_idx, u32 var_ref_data) noexcept;
  /// Init a variable-ref assignment
  void init_variable_ref(IRValueRef value, u32 var_ref_data) noexcept {
    init_variable_ref(adaptor->val_local_idx(value), var_ref_data);
  }

  i32 allocate_stack_slot(u32 size) noexcept;
  void free_stack_slot(u32 slot, u32 size) noexcept;

  template <typename Fn>
  void handle_func_arg(u32 arg_idx, IRValueRef arg, Fn add_arg) noexcept;

  ValueRef val_ref(IRValueRef value) noexcept;

  std::pair<ValueRef, ValuePartRef> val_ref_single(IRValueRef value) noexcept;

  /// Get a defining reference to a value
  ValueRef result_ref(IRValueRef value) noexcept;

  std::pair<ValueRef, ValuePartRef>
      result_ref_single(IRValueRef value) noexcept;

  void set_value(ValuePartRef &val_ref, ScratchReg &scratch) noexcept;
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

  /// Reload a value part from memory or recompute variable address.
  void reload_to_reg(AsmReg dst, AssignmentPartRef ap) noexcept;

  void allocate_spill_slot(AssignmentPartRef ap) noexcept;

  /// Ensure the value is spilled in its stack slot (except variable refs).
  void spill(AssignmentPartRef ap) noexcept;

  /// Evict the value from its register, spilling if needed, and free register.
  void evict(AssignmentPartRef ap) noexcept;

  /// Evict the value from the register, spilling if needed, and free register.
  void evict_reg(Reg reg) noexcept;

  /// Free the register. Requires that the contained value is already spilled.
  void free_reg(Reg reg) noexcept;

  // TODO(ts): switch to a branch_spill_before naming style?
  typename RegisterFile::RegBitSet
      spill_before_branch(bool force_spill = false) noexcept;
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

#ifndef NDEBUG
  bool may_change_value_state() const noexcept { return !generating_branch; }
#endif

  void move_to_phi_nodes(BlockIndex target) noexcept {
    if (analyzer.block_has_phis(target)) {
      move_to_phi_nodes_impl(target);
    }
  }

  void move_to_phi_nodes_impl(BlockIndex target) noexcept;

  bool branch_needs_split(IRBlockRef target) noexcept {
    // for now, if the target has PHI-nodes, we split
    return analyzer.block_has_phis(target);
  }

  BlockIndex next_block() const noexcept;

  bool try_force_fixed_assignment(IRValueRef) const noexcept { return false; }

  bool hook_post_func_sym_init() noexcept { return true; }

  void analysis_start() noexcept {}

  void analysis_end() noexcept {}

  void reloc_text(Assembler::SymRef sym,
                  u32 type,
                  u64 offset,
                  i64 addend = 0) noexcept {
    this->assembler.reloc_sec(
        text_writer.get_sec_ref(), sym, type, offset, addend);
  }

  void label_place(Assembler::Label label) noexcept {
    this->assembler.label_place(
        label, text_writer.get_sec_ref(), text_writer.offset());
  }

protected:
  Assembler::SymRef get_personality_sym() noexcept;

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
  cca.bank = vp.bank();
  cca.size = vp.part_size();

  assigner.assign_arg(cca);

  if (cca.byval) {
    derived()->add_arg_byval(vp, cca);
    vp.reset(&compiler);
  } else if (!cca.reg.valid()) {
    derived()->add_arg_stack(vp, cca);
    vp.reset(&compiler);
  } else {
    u32 size = vp.part_size();
    if (vp.is_in_reg(cca.reg)) {
      if (!vp.can_salvage()) {
        compiler.evict_reg(cca.reg);
      } else {
        vp.salvage(&compiler);
      }
      if (cca.sext || cca.zext) {
        compiler.generate_raw_intext(cca.reg, cca.reg, cca.sext, 8 * size, 64);
      }
    } else {
      if (compiler.register_file.is_used(cca.reg)) {
        compiler.evict_reg(cca.reg);
      }
      if (vp.can_salvage()) {
        AsmReg vp_reg = vp.salvage(&compiler);
        if (cca.sext || cca.zext) {
          compiler.generate_raw_intext(cca.reg, vp_reg, cca.sext, 8 * size, 64);
        } else {
          compiler.mov(cca.reg, vp_reg, size);
        }
      } else {
        vp.reload_into_specific_fixed(&compiler, cca.reg);
        if (cca.sext || cca.zext) {
          compiler.generate_raw_intext(
              cca.reg, cca.reg, cca.sext, 8 * size, 64);
        }
      }
    }
    vp.reset(&compiler);
    assert(!compiler.register_file.is_used(cca.reg));
    compiler.register_file.mark_used(
        cca.reg, CompilerBase::INVALID_VAL_LOCAL_IDX, 0);
    compiler.register_file.mark_fixed(cca.reg);
    compiler.register_file.mark_clobbered(cca.reg);
    arg_regs |= (1ull << cca.reg.id());
  }
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
template <typename CBDerived>
void CompilerBase<Adaptor, Derived, Config>::CallBuilderBase<
    CBDerived>::add_arg(CallArg &&arg) noexcept {
  ValueRef vr = compiler.val_ref(arg.value);
  const u32 part_count = compiler.val_parts(arg.value).count();

  if (arg.flag == CallArg::Flag::byval) {
    assert(part_count == 1);
    add_arg(vr.part(0),
            CCAssignment{
                .byval = true,
                .byval_align = arg.byval_align,
                .byval_size = arg.byval_size,
            });
    return;
  }

  u32 align = 1;
  bool consecutive = false;
  u32 consec_def = 0;
  if (compiler.arg_is_int128(arg.value)) {
    // TODO: this also applies to composites with 16-byte alignment
    align = 16;
    consecutive = true;
  } else if (part_count > 1 &&
             !compiler.arg_allow_split_reg_stack_passing(arg.value)) {
    consecutive = true;
    if (part_count > UINT8_MAX) {
      // Must be completely passed on the stack.
      consecutive = false;
      consec_def = -1;
    }
  }

  for (u32 part_idx = 0; part_idx < part_count; ++part_idx) {
    derived()->add_arg(
        vr.part(part_idx),
        CCAssignment{
            .consecutive =
                u8(consecutive ? part_count - part_idx - 1 : consec_def),
            .sret = arg.flag == CallArg::Flag::sret,
            .sext = arg.flag == CallArg::Flag::sext,
            .zext = arg.flag == CallArg::Flag::zext,
            .align = u8(part_idx == 0 ? align : 1),
        });
  }
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
template <typename CBDerived>
void CompilerBase<Adaptor, Derived, Config>::CallBuilderBase<CBDerived>::call(
    std::variant<typename Assembler::SymRef, ValuePart> target) noexcept {
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

  for (auto reg_id : util::BitSetIterator<>{arg_regs}) {
    compiler.register_file.unmark_fixed(Reg{reg_id});
    compiler.register_file.unmark_used(Reg{reg_id});
  }
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

  if (vp.is_in_reg(cca.reg)) {
    if (!vp.can_salvage()) {
      compiler.evict_reg(cca.reg);
    } else {
      vp.salvage(&compiler);
    }
    if (cca.sext || cca.zext) {
      compiler.generate_raw_intext(cca.reg, cca.reg, cca.sext, 8 * size, 64);
    }
  } else {
    if (compiler.register_file.is_used(cca.reg)) {
      compiler.evict_reg(cca.reg);
    }
    if (vp.can_salvage()) {
      AsmReg vp_reg = vp.salvage(&compiler);
      if (cca.sext || cca.zext) {
        compiler.generate_raw_intext(cca.reg, vp_reg, cca.sext, 8 * size, 64);
      } else {
        compiler.mov(cca.reg, vp_reg, size);
      }
    } else {
      vp.reload_into_specific_fixed(&compiler, cca.reg);
      if (cca.sext || cca.zext) {
        compiler.generate_raw_intext(cca.reg, cca.reg, cca.sext, 8 * size, 64);
      }
    }
  }
  vp.reset(&compiler);
  assert(!compiler.register_file.is_used(cca.reg));
  compiler.register_file.mark_used(
      cca.reg, CompilerBase::INVALID_VAL_LOCAL_IDX, 0);
  compiler.register_file.mark_fixed(cca.reg);
  compiler.register_file.mark_clobbered(cca.reg);
  ret_regs |= (1ull << cca.reg.id());
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
void CompilerBase<Adaptor, Derived, Config>::RetBuilder::add(
    IRValueRef val, CallArg::Flag flag) noexcept {
  u32 part_count = compiler.val_parts(val).count();
  ValueRef vr = compiler.val_ref(val);
  for (u32 part_idx = 0; part_idx < part_count; ++part_idx) {
    add(vr.part(part_idx),
        CCAssignment{
            .sext = flag == CallArg::Flag::sext,
            .zext = flag == CallArg::Flag::zext,
        });
  }
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
void CompilerBase<Adaptor, Derived, Config>::RetBuilder::ret() noexcept {
  for (auto reg_id : util::BitSetIterator<>{ret_regs}) {
    compiler.register_file.unmark_fixed(Reg{reg_id});
    compiler.register_file.unmark_used(Reg{reg_id});
  }

  compiler.gen_func_epilog();
  compiler.release_regs_after_return();
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
bool CompilerBase<Adaptor, Derived, Config>::compile() {
  // create function symbols
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
      AsmReg reg = derived()->select_fixed_assignment_reg(ap.bank(), value);
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
#ifndef NDEBUG
  assignment->pending_free = false;
#endif
  assignment->variable_ref = false;
  assignment->stack_variable = false;
  assignment->delay_free = last_full;
  assignment->part_count = part_count;
  assignment->frame_off = 0;
  assignment->references_left = ref_count;
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
void CompilerBase<Adaptor, Derived, Config>::free_assignment(
    const ValLocalIdx local_idx) noexcept {
  TPDE_LOG_TRACE("Freeing assignment for value {}",
                 static_cast<u32>(local_idx));

  ValueAssignment *assignment =
      assignments.value_ptrs[static_cast<u32>(local_idx)];
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
  if (!is_var_ref && assignment->frame_off != 0) {
    free_stack_slot(assignment->frame_off, assignment->size());
  }

  assignments.allocator.deallocate(assignment);
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
template <typename Fn>
void CompilerBase<Adaptor, Derived, Config>::handle_func_arg(
    u32 arg_idx, IRValueRef arg, Fn add_arg) noexcept {
  ValueRef vr = derived()->result_ref(arg);
  if (adaptor->cur_arg_is_byval(arg_idx)) {
    add_arg(vr.part(0),
            CCAssignment{
                .byval = true,
                .byval_align = adaptor->cur_arg_byval_align(arg_idx),
                .byval_size = adaptor->cur_arg_byval_size(arg_idx),
            });
    return;
  }

  if (adaptor->cur_arg_is_sret(arg_idx)) {
    add_arg(vr.part(0), CCAssignment{.sret = true});
    return;
  }

  const u32 part_count = derived()->val_parts(arg).count();

  u32 align = 1;
  u32 consecutive = 0;
  u32 consec_def = 0;
  if (derived()->arg_is_int128(arg)) {
    // TODO: this also applies to composites with 16-byte alignment
    align = 16;
    consecutive = 1;
  } else if (part_count > 1 &&
             !derived()->arg_allow_split_reg_stack_passing(arg)) {
    consecutive = 1;
    if (part_count > UINT8_MAX) {
      // Must be completely passed on the stack.
      consecutive = 0;
      consec_def = -1;
    }
  }

  for (u32 part_idx = 0; part_idx < part_count; ++part_idx) {
    add_arg(vr.part(part_idx),
            CCAssignment{
                .consecutive =
                    u8(consecutive ? part_count - part_idx - 1 : consec_def),
                .align = u8(part_idx == 0 ? align : 1),
            });
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
void CompilerBase<Adaptor, Derived, Config>::set_value(
    ValuePartRef &val_ref, ScratchReg &scratch) noexcept {
  auto ap = val_ref.assignment();
  assert(scratch.has_reg());
  auto reg = scratch.cur_reg();

  if (ap.fixed_assignment()) {
    auto cur_reg = ap.get_reg();
    assert(register_file.is_used(cur_reg));
    assert(register_file.is_fixed(cur_reg));
    assert(register_file.reg_local_idx(cur_reg) == val_ref.local_idx());

    if (cur_reg.id() != reg.id()) {
      derived()->mov(cur_reg, reg, ap.part_size());
    }

    ap.set_register_valid(true);
    ap.set_modified(true);
    return;
  }

  if (ap.register_valid()) {
    auto cur_reg = ap.get_reg();
    if (cur_reg.id() == reg.id()) {
      ap.set_modified(true);
      return;
    }
    val_ref.unlock();
    assert(!register_file.is_fixed(cur_reg));
    register_file.unmark_used(cur_reg);
  }

  // ScratchReg's reg is fixed and used => unfix, keep used, update assignment
  assert(register_file.is_used(reg));
  assert(register_file.is_fixed(reg));
  assert(register_file.is_clobbered(reg));
  scratch.force_set_reg(AsmReg::make_invalid());
  register_file.unmark_fixed(reg);
  register_file.update_reg_assignment(reg, val_ref.local_idx(), val_ref.part());
  ap.set_reg(reg);
  ap.set_register_valid(true);
  ap.set_modified(true);
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
  // x64) and possible compile time as there might be additional logic to move
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

  bool must_spill = force_spill;
  if (!must_spill) {
    // We must always spill if no block is immediately succeeding or that block
    // has multiple incoming edges.
    auto next_block_is_succ = false;
    auto next_block_has_multiple_incoming = false;
    u32 succ_count = 0;
    for (const IRBlockRef succ : adaptor->block_succs(cur_block_ref)) {
      ++succ_count;
      if (static_cast<u32>(analyzer.block_idx(succ)) ==
          static_cast<u32>(cur_block_idx) + 1) {
        next_block_is_succ = true;
        if (analyzer.block_has_multiple_incoming(succ)) {
          next_block_has_multiple_incoming = true;
        }
      }
    }

    must_spill = !next_block_is_succ || next_block_has_multiple_incoming;

    if (succ_count == 1 && !must_spill) {
      return RegBitSet{};
    }
  }

  /*if (!next_block_is_succ) {
      // spill and remove from reg assignment
      auto spilled = RegisterFile::RegBitSet{};
      for (auto reg : register_file.used_regs()) {
          auto local_idx = register_file.reg_local_idx(AsmReg{reg});
          auto part      = register_file.reg_part(AsmReg{reg});
          if (local_idx == INVALID_VAL_LOCAL_IDX) {
              // TODO(ts): can this actually happen?
              continue;
          }
          auto ap = AssignmentPartRef{val_assignment(local_idx), part};
          ap.spill_if_needed(this);
          spilled |= RegisterFile::RegBitSet{1ull} << reg;
      }
      return spilled;
  }*/

  const auto spill_reg_if_needed = [&](const auto reg) {
    auto local_idx = register_file.reg_local_idx(AsmReg{reg});
    auto part = register_file.reg_part(AsmReg{reg});
    if (local_idx == INVALID_VAL_LOCAL_IDX) {
      // scratch regs can never be held across blocks
      return true;
    }
    auto *assignment = val_assignment(local_idx);
    auto ap = AssignmentPartRef{assignment, part};
    if (ap.fixed_assignment()) {
      // fixed registers do not need to be spilled
      return true;
    }

    if (!ap.modified()) {
      // no need to spill if the value was not modified
      return false;
    }

    if (ap.variable_ref()) {
      return false;
    }

    const auto &liveness = analyzer.liveness_info(local_idx);
    if (liveness.last <= cur_block_idx) {
      // no need to spill value if it dies immediately after the block
      return false;
    }

    if (must_spill) {
      spill(ap);
      return false;
    }

    // spill if the value is alive in any successor that is not the next
    // block
    for (const IRBlockRef succ : adaptor->block_succs(cur_block_ref)) {
      const auto block_idx = analyzer.block_idx(succ);
      if (static_cast<u32>(block_idx) == static_cast<u32>(cur_block_idx) + 1) {
        continue;
      }
      if (block_idx >= liveness.first && block_idx <= liveness.last) {
        spill(ap);
        return false;
      }
    }

    return false;
  };

  auto spilled = RegBitSet{};
  // TODO(ts): just use register_file.used_nonfixed_regs()?
  for (auto reg : register_file.used_regs()) {
    const auto reg_fixed = spill_reg_if_needed(reg);
    // remove from register assignment if the next block cannot rely on the
    // value being in the specific register
    if (!reg_fixed && must_spill) {
      // TODO(ts): this needs to be changed if this is supposed to work
      // with other RegisterFile implementations
      spilled |= RegBitSet{1ull} << reg;
    }
  }

  return spilled;
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
void CompilerBase<Adaptor, Derived, Config>::release_spilled_regs(
    typename RegisterFile::RegBitSet regs) noexcept {
  assert(may_change_value_state());

  // TODO(ts): needs changes for other RegisterFile impls
  for (auto reg_id : util::BitSetIterator<>{regs}) {
    auto reg = AsmReg{reg_id};
    if (!register_file.is_used(reg)) {
      continue;
    }
    if (register_file.is_fixed(reg)) {
      // we don't need to release fixed assignments but they should be
      // real assignments from values
      assert(register_file.reg_local_idx(reg) != INVALID_VAL_LOCAL_IDX);
      continue;
    }

    free_reg(reg);
  }
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
void CompilerBase<Adaptor, Derived, Config>::
    release_regs_after_return() noexcept {
  // we essentially have to free all non-fixed registers
  for (auto reg_id : register_file.used_nonfixed_regs()) {
    free_reg(Reg{reg_id});
  }
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
    // TODO(ts): if phi==incoming_val, we should be able to elide the move
    // even if the phi is in a fixed register, no?

    auto phi_vr = derived()->result_ref(phi);
    auto val_vr = derived()->val_ref(incoming_val);
    if (phi == incoming_val) {
      return;
    }

    u32 part_count = phi_vr.assignment()->part_count;
    for (u32 i = 0; i < part_count; ++i) {
      AssignmentPartRef phi_ap{phi_vr.assignment(), i};

      AsmReg reg{};
      ValuePartRef val_vpr = val_vr.part(i);
      if (val_vpr.is_const()) {
        reg = scratch.alloc_from_bank(val_vpr.bank());
        val_vpr.reload_into_specific_fixed(reg);
      } else if (val_vpr.assignment().register_valid() ||
                 val_vpr.assignment().fixed_assignment()) {
        reg = val_vpr.assignment().get_reg();
      } else {
        reg = val_vpr.reload_into_specific_fixed(
            this, scratch.alloc_from_bank(phi_ap.bank()));
      }

      if (phi_ap.fixed_assignment()) {
        derived()->mov(phi_ap.get_reg(), reg, phi_ap.part_size());
      } else {
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
typename CompilerBase<Adaptor, Derived, Config>::Assembler::SymRef
    CompilerBase<Adaptor, Derived, Config>::get_personality_sym() noexcept {
  using SymRef = typename Assembler::SymRef;
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
  this->text_writer.growth_size = expected_code_size;
  this->text_writer.ensure_space(expected_code_size);

  derived()->start_func(func_idx);

  block_labels.clear();
  block_labels.resize_uninitialized(analyzer.block_layout.size());
  for (u32 i = 0; i < analyzer.block_layout.size(); ++i) {
    block_labels[i] = assembler.label_create();
  }

  // TODO(ts): place function label
  // TODO(ts): make function labels optional?

  CCAssigner *cc_assigner = derived()->cur_cc_assigner();
  assert(cc_assigner != nullptr);

  register_file.allocatable = cc_assigner->get_ccinfo().allocatable_regs;

  // This initializes the stack frame, which must reserve space for
  // callee-saved registers, vararg save area, etc.
  cc_assigner->reset();
  derived()->gen_func_prolog_and_args(cc_assigner);

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

  if (static_cast<u32>(assignments.delayed_free_lists[block_idx]) != ~0u) {
    auto list_entry = assignments.delayed_free_lists[block_idx];
    while (static_cast<u32>(list_entry) != ~0u) {
      auto next_entry = assignments.value_ptrs[static_cast<u32>(list_entry)]
                            ->next_delayed_free_entry;
      derived()->free_assignment(list_entry);
      list_entry = next_entry;
    }
  }
  return true;
}

} // namespace tpde
