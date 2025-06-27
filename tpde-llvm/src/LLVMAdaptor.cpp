// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "LLVMAdaptor.hpp"

#include <llvm/ADT/STLExtras.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/ReplaceConstant.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/TimeProfiler.h>
#include <llvm/Support/raw_ostream.h>

#include <algorithm>
#include <utility>

#include "base.hpp"
#include "tpde/base.hpp"
#include "tpde/util/misc.hpp"

namespace tpde_llvm {

static void sort_phi_entries(llvm::PHINode *phi) {
  unsigned count = phi->getNumIncomingValues();
  auto blocks = phi->block_begin();
  auto values = phi->op_begin();
  auto swap = [phi, blocks, values](unsigned a, unsigned b) {
    values[a].swap(values[b]);
    llvm::BasicBlock *tmp = blocks[a];
    phi->setIncomingBlock(a, blocks[b]);
    phi->setIncomingBlock(b, tmp);
  };

  // Simple heap sort
  unsigned start = count / 2;
  unsigned end = count;
  while (end > 1) {
    if (start) {
      --start;
    } else {
      --end;
      swap(0, end);
    }
    unsigned root = start;
    while (2 * root + 1 < end) {
      unsigned child = 2 * root + 1;
      if (child + 1 < end && blocks[child] < blocks[child + 1]) {
        child = child + 1;
      }
      if (blocks[root] < blocks[child]) {
        swap(root, child);
        root = child;
      } else {
        break;
      }
    }
  }

  assert(std::is_sorted(blocks, blocks + count));
}

std::pair<llvm::Value *, llvm::Instruction *>
    LLVMAdaptor::fixup_constant(llvm::Constant *cst,
                                llvm::Instruction *ins_before) {
  if (auto *cexpr = llvm::dyn_cast<llvm::ConstantExpr>(cst)) [[unlikely]] {
    // clang creates these and we don't want to handle them so
    // we split them up into their own values
    llvm::Instruction *expr_inst = cexpr->getAsInstruction();
    expr_inst->insertBefore(ins_before);
    return {expr_inst, expr_inst};
  }

  if (auto *cv = llvm::dyn_cast<llvm::ConstantVector>(cst)) [[unlikely]] {
    // TODO: optimize so that all supported constants are in a top-level
    // ConstantDataVector. E.g., <poison, 0> could be replaced with zero.
    llvm::Instruction *ins_begin = nullptr;
    llvm::Type *el_ty = cv->getType()->getScalarType();
    llvm::Constant *el_zero = llvm::Constant::getNullValue(el_ty);

    llvm::SmallVector<llvm::Constant *> base;
    llvm::SmallVector<llvm::Value *> repls;
    for (auto it : llvm::enumerate(cv->operands())) {
      auto *cst = llvm::cast<llvm::Constant>(it.value());
      if (llvm::isa<llvm::UndefValue, llvm::PoisonValue>(cst)) {
        // replace undef/poison with zero
        base.push_back(el_zero);
      } else if (llvm::isa<llvm::ConstantData>(cst)) {
        // other ConstantData (null pointer, int, fp) is fine
        base.push_back(cst);
      } else {
        assert((llvm::isa<llvm::GlobalValue, llvm::ConstantExpr>(cst)) &&
               "unexpected constant type in vector");
        base.push_back(el_zero);
        repls.resize(cv->getNumOperands());
        if (auto [repl, inst] = fixup_constant(cst, ins_before); repl) {
          repls[it.index()] = repl;
          if (!ins_begin) {
            ins_begin = inst;
          }
        } else {
          repls[it.index()] = cst;
        }
      }
    }

    llvm::Value *repl = llvm::ConstantVector::get(base);
    // NB: this is likely a ConstantDataSequential, but could also be a
    // ConstantAggregateZero or still a ConstantVector for weird types.
    if (repls.empty()) {
      return {repl, nullptr};
    }

    llvm::Type *i32 = llvm::Type::getInt32Ty(*context);
    for (auto it : llvm::enumerate(repls)) {
      auto *el = it.value() ? it.value() : cv->getOperand(it.index());
      auto *idx_val = llvm::ConstantInt::get(i32, it.index());
      repl = llvm::InsertElementInst::Create(repl, el, idx_val, "", ins_before);
      if (!ins_begin) {
        ins_begin = llvm::cast<llvm::Instruction>(repl);
      }
    }
    return {repl, ins_begin};
  }

  if (auto *agg = llvm::dyn_cast<llvm::ConstantAggregate>(cst)) [[unlikely]] {
    llvm::Instruction *ins_begin = nullptr;
    llvm::SmallVector<llvm::Value *> repls;
    for (auto it : llvm::enumerate(agg->operands())) {
      auto *cst = llvm::cast<llvm::Constant>(it.value());
      if (auto [repl, inst] = fixup_constant(cst, ins_before); repl) {
        repls.resize(agg->getNumOperands());
        repls[it.index()] = repl;
        if (!ins_begin) {
          ins_begin = inst;
        }
      }
    }
    if (!repls.empty()) {
      // TODO: optimize so that all supported constants are in the
      // top-level constant?
      llvm::Value *repl = llvm::PoisonValue::get(cst->getType());
      for (auto it : llvm::enumerate(repls)) {
        unsigned idx = it.index();
        auto *el = it.value() ? it.value() : agg->getOperand(idx);
        repl = llvm::InsertValueInst::Create(repl, el, {idx}, "", ins_before);
      }
      return {repl, ins_begin};
    }
  }

  return {nullptr, nullptr};
}

llvm::Instruction *LLVMAdaptor::handle_inst_in_block(llvm::Instruction *inst) {
  llvm::Instruction *restart_from = nullptr;

  // TODO: remove this hack, see compile_invoke.
  if (llvm::isa<llvm::InvokeInst>(inst)) {
    func_has_dynamic_alloca = true;
  }

  // Check operands for constants; PHI nodes are handled by predecessors.
  if (!llvm::isa<llvm::PHINode>(inst)) {
    for (llvm::Use &use : inst->operands()) {
      if (!llvm::isa<llvm::ConstantAggregate, llvm::ConstantExpr>(use.get())) {
        continue;
      }

      auto *cst = llvm::cast<llvm::Constant>(use.get());
      if (auto [repl, ins_begin] = fixup_constant(cst, inst); repl)
          [[unlikely]] {
        use = repl;
        if (!restart_from) {
          restart_from = ins_begin;
        }
      }
    }
  }

  auto fused = false;
  if (const auto *alloca = llvm::dyn_cast<llvm::AllocaInst>(inst)) {
    if (is_static_alloca(alloca)) {
      initial_stack_slot_indices.push_back(alloca);
      // fuse static alloca's in the initial block so we dont try
      // to dynamically allocate them
      fused = true;
    } else {
      func_has_dynamic_alloca = true;
    }
  }

  if (restart_from) {
    return restart_from;
  }

  auto val_idx = values.size();
  val_idx_for_inst(inst) = val_idx;

#ifndef NDEBUG
  assert(!value_lookup.contains(inst));
  value_lookup.insert_or_assign(inst, val_idx);
#endif
  auto [ty, complex_part_idx] = lower_type(inst->getType());
  values.push_back(ValInfo{
      .type = ty, .fused = fused, .complex_part_tys_idx = complex_part_idx});
  return nullptr;
}

bool LLVMAdaptor::switch_func(const IRFuncRef function) noexcept {
  llvm::TimeTraceScope time_scope("TPDE_Prepass", [function]() {
    // getName is expensive, so only call it when time tracing is enabled.
    return std::string(function->getName());
  });

  cur_func = function;
  func_unsupported = false;

  TPDE_LOG_DBG("Compiling func: {}",
               static_cast<std::string_view>(function->getName()));

#ifndef NDEBUG
  value_lookup.clear();
  #if LLVM_VERSION_MAJOR < 20
  block_lookup.clear();
  #endif
#endif
  blocks.clear();
  block_succ_indices.clear();
  block_succ_ranges.clear();
  initial_stack_slot_indices.clear();
  func_has_dynamic_alloca = false;

  // we keep globals around for all function compilation
  // and assign their value indices at the start of the compilation
  // TODO(ts): move this to start_compile?
  if (!globals_init) {
    globals_init = true;
    global_lookup.reserve(512);
    global_list.reserve(512);
    auto add_global = [&](llvm::GlobalValue *gv) {
      assert(global_lookup.find(gv) == global_lookup.end());
      assert(global_list.size() == values.size());
      global_list.push_back(gv);
      global_lookup.insert_or_assign(gv, values.size());
      values.push_back(ValInfo{.type = LLVMBasicValType::ptr,
                               .fused = false,
                               .complex_part_tys_idx = ~0u});

      if (gv->isThreadLocal()) [[unlikely]] {
        // Rewrite all accesses to thread-local variables to go through the
        // intrinsic llvm.threadlocal.address; other accesses are unsupported.
        auto handle_thread_local_uses = [](llvm::GlobalValue *gv) -> bool {
          for (llvm::Use &use : llvm::make_early_inc_range(gv->uses())) {
            llvm::User *user = use.getUser();
            auto *intrin = llvm::dyn_cast<llvm::IntrinsicInst>(user);
            if (intrin && intrin->getIntrinsicID() ==
                              llvm::Intrinsic::threadlocal_address) {
              continue;
            }

            auto *instr = llvm::dyn_cast<llvm::Instruction>(user);
            if (!instr) [[unlikely]] {
              return false;
            }

            llvm::IRBuilder<> irb(instr);
            use.set(irb.CreateThreadLocalAddress(use.get()));
          }
          return true;
        };

        // We do two passes. The first pass handle the common case, which is
        // that all users are instructions. For ConstantExpr users (e.g., GEP),
        // we do a second pass after expanding these (which is expensive).
        if (!handle_thread_local_uses(gv)) {
          llvm::convertUsersOfConstantsToInstructions(gv);
          if (!handle_thread_local_uses(gv)) {
            TPDE_LOG_ERR("thread-local global with unsupported uses");
            for (llvm::Use &use : gv->uses()) {
              std::string user;
              llvm::raw_string_ostream(user) << *use.getUser();
              TPDE_LOG_INFO("use: {}", user);
            }
            func_unsupported = true;
          }
        }
      }
    };
    for (llvm::GlobalVariable &gv : mod->globals()) {
      add_global(&gv);
    }
    for (llvm::GlobalAlias &ga : mod->aliases()) {
      add_global(&ga);
    }
    // Do functions last, handling of global variables/aliases might introduce
    // another intrinsic declaration for llvm.threadlocal.address.
    for (llvm::Function &fn : mod->functions()) {
      add_global(&fn);
    }
    global_idx_end = values.size();
  } else {
    values.resize(global_idx_end);
  }

  const size_t arg_count = function->arg_size();
  for (size_t i = 0; i < arg_count; ++i) {
    llvm::Argument *arg = function->getArg(i);
    const auto [ty, complex_part_idx] = lower_type(arg->getType());
    values.push_back(ValInfo{
        .type = ty, .fused = false, .complex_part_tys_idx = complex_part_idx});

    // Check that all parameter types are layout-compatible to LLVM.
    check_type_compatibility(arg->getType(), ty, complex_part_idx);
  }

  // Check that the return type is layout-compatible to LLVM.
  check_type_compatibility(function->getReturnType());

#if LLVM_VERSION_MAJOR >= 20
  // Renumber blocks in the order they occur in the function.
  function->renumberBlocks();
  blocks.reserve(function->getMaxBlockNumber());
#endif

  for (llvm::BasicBlock &block : *function) {
    auto it = block.begin(), end = block.end();
    // In the first pass, fixup all constants in phis. This might insert
    // instructions into all predecessors.
    while (it != end && llvm::isa<llvm::PHINode>(*it)) {
      llvm::PHINode &phi = llvm::cast<llvm::PHINode>(*it);
      auto blocks = phi.block_begin();
      auto values = phi.op_begin();
      unsigned num_incoming = phi.getNumIncomingValues();
      for (unsigned i = 0; i < num_incoming; ++i) {
        llvm::Value *val = values[i];
        llvm::BasicBlock *block = blocks[i];
        if (llvm::isa<llvm::ConstantAggregate, llvm::ConstantExpr>(val)) {
          auto *ins_before = block->getTerminator();
          if (block->begin().getNodePtr() != ins_before) {
            auto *prev_inst = ins_before->getPrevNonDebugInstruction();
            if (prev_inst && llvm::isa<llvm::CmpInst>(prev_inst)) {
              // make sure fusing can still happen
              ins_before = prev_inst;
            }
          }

          auto *cst = llvm::cast<llvm::Constant>(val);
          if (auto [repl, _] = fixup_constant(cst, ins_before); repl) {
            // This replaces *all* values for multi-edges.
            phi.setIncomingValueForBlock(block, repl);
          }
        }
      }

      if (num_incoming >= PHINodeSortThreshold) [[unlikely]] {
        sort_phi_entries(&phi);
      }

      ++it;
    }

    // Here, store iterator of last phi (if any). We cannot store an iterator of
    // the next instruction, as another block's phi node might cause
    // instructions to be inserted immediately after the last phi node.
    if (it != block.begin()) {
      --it;
    } else {
      it = block.end();
    }

#if LLVM_VERSION_MAJOR >= 20
    assert(block.getNumber() == blocks.size());
#else
  #ifndef NDEBUG
    block_lookup[&block] = blocks.size();
  #endif
    block_embedded_idx(&block) = blocks.size();
#endif

    blocks.push_back(
        BlockInfo{.block = &block, .aux = BlockAux{.phi_end = it}});
  }

  for (BlockInfo &info : blocks) {
    llvm::BasicBlock *block = info.block;
    for (auto it = block->begin(), end = block->end(); it != end;) {
      auto *restart_from = handle_inst_in_block(&*it);
      if (restart_from) {
        it = restart_from->getIterator();
      } else {
        ++it;
      }
    }

    // phi_end must point to the instruction *after* the last phi (so that begin
    // ()..phi_end described all phis). We need these hacks, as we might insert
    // instructions immediately after the phi node.
    if (info.aux.phi_end == block->end()) {
      info.aux.phi_end = block->begin();
    } else {
      ++info.aux.phi_end; // phi_end points to the instr after the phi again
    }

    const u32 start_idx = block_succ_indices.size();
    for (auto *succ : llvm::successors(info.block)) {
      block_succ_indices.push_back(block_lookup_idx(succ));
    }
    block_succ_ranges.push_back(
        std::make_pair(start_idx, block_succ_indices.size()));
  }

  return !func_unsupported;
}

void LLVMAdaptor::switch_module(llvm::Module &mod) noexcept {
  if (this->mod) {
    reset();
  }
  this->context = &mod.getContext();
  this->mod = &mod;
  this->mod->setDataLayout(data_layout);
}

void LLVMAdaptor::reset() noexcept {
  context = nullptr;
  mod = nullptr;
  values.clear();
  global_lookup.clear();
  global_list.clear();
#ifndef NDEBUG
  value_lookup.clear();
  #if LLVM_VERSION_MAJOR < 20
  block_lookup.clear();
  #endif
#endif
  complex_part_types.clear();
  complex_type_map.clear();
  initial_stack_slot_indices.clear();
  cur_func = nullptr;
  globals_init = false;
  global_idx_end = 0;
  blocks.clear();
  block_succ_indices.clear();
  block_succ_ranges.clear();
}

void LLVMAdaptor::report_incompatible_type(llvm::Type *type) noexcept {
  std::string type_name;
  llvm::raw_string_ostream(type_name) << *type;
  TPDE_LOG_ERR("type with incompatible layout at function/call: {}", type_name);
  func_unsupported = true;
}

void LLVMAdaptor::report_unsupported_type(llvm::Type *type) noexcept {
  std::string type_name;
  llvm::raw_string_ostream(type_name) << *type;
  TPDE_LOG_ERR("unsupported type: {}", type_name);
  func_unsupported = true;
}

static std::tuple<LLVMBasicValType, u32, bool>
    lower_vector_type(const llvm::FixedVectorType *type) noexcept {
  auto *el_ty = type->getElementType();
  auto num_elts = type->getNumElements();

  // LLVM vectors have two different representations, the in-memory/bitcast
  // representation and the in-register representation. For types that are not
  // directly legal, these are not equivalent.
  //
  // The in-memory type is always dense, i.e. <N x iM> is like an i(N*M),
  // and only unspecified when N*M is not a multiple of 8.
  //
  // The in-register type, which is also used for passing parameters, is
  // legalized as follows (see TargetLoweringBase::getTypeConversion):
  // - Target-specific legalization rules are applied.
  //   - E.g., AArch64: widen v1i16 => v4i16 instead of scalarizing
  //     (see AArch64TargetLowering::getPreferredVectorAction)
  // - If single element, scalarize.
  // - Widen number of elements to next power of two.
  // - For integer types: increase width until legal type is found.
  //   - E.g., AArch64: v2i16 => v2i32 (ISA supports 2s) (final result)
  //   - E.g., x86-64: v2i16 unchanged (ISA supports neither v2i32, v2i64)
  // - Widen number of elements in powers of two until legal type is found.
  //   - E.g., x86-64: v2i16 => v8i16 (final result)
  // - If type still not legal, split in half and repeat.
  //
  // This also handles illegal types. E.g., v3i99 would first get widened to
  // v4i99, then (as no i99 is legal anywhere) split into v2i99, which is
  // legalized recursively; this gets split into v1i99; this gets scalarized
  // into i99; this gets promoted into i128 (next power of two); this gets
  // expanded into two i64 (as i128 is not legal for any register class).
  //
  //
  // To avoid the difficulties of legalizing all kinds of vector types, we
  // only support types that are directly legal or can be legalized by
  // just widening (e.g., v2f32 => v4f32) on x86-64 *and* AArch64 for now.
  //
  // Therefore, we support:
  // - Integer elements: v8i8/v16i8, v4i16/v8i16, v2i32/v4i32, v2i64
  //   (no widened vectors: x86-64 would widen, but AArch64 would promote)
  // - Floating-point elements: v2f32, v4f32, v2f64
  //   (single-element vectors would be scalarized; v3f32 would need 12b load)
  // Other vector types are scalarized.
  //
  // We can only handle types where the compact bitwise representation is the
  // same as the array representation. Otherwise, load/store would need to
  // decompress/compress vectors.
  switch (el_ty->getTypeID()) {
  case llvm::Type::IntegerTyID: {
    unsigned el_width = el_ty->getIntegerBitWidth();
    if (el_width & (el_width - 1)) {
      return {LLVMBasicValType::invalid, 0, false};
    }
    if (el_width == 1) {
      if (num_elts > 64) {
        return {LLVMBasicValType::invalid, 0, false};
      }
      constexpr unsigned base = unsigned(LLVMBasicValType::v8i1);
      static_assert(base + 1 == unsigned(LLVMBasicValType::v16i1));
      static_assert(base + 2 == unsigned(LLVMBasicValType::v32i1));
      static_assert(base + 3 == unsigned(LLVMBasicValType::v64i1));

      unsigned off = 31 - tpde::util::cnt_lz((num_elts - 1) >> 2 | 1);
      return {static_cast<LLVMBasicValType>(base + off), 1, false};
    }
    switch (tpde::util::cnt_tz(el_width) - 3) {
    case 0:
      if (num_elts % 16 == 0) {
        return {LLVMBasicValType::v16i8, num_elts / 16, num_elts != 16};
      } else if (num_elts % 8 == 0) {
        return {LLVMBasicValType::v8i8, num_elts / 8, num_elts != 8};
      }
      return {LLVMBasicValType::i8, num_elts, true};
    case 1:
      if (num_elts % 8 == 0) {
        return {LLVMBasicValType::v8i16, num_elts / 8, num_elts != 8};
      } else if (num_elts % 4 == 0) {
        return {LLVMBasicValType::v4i16, num_elts / 4, num_elts != 4};
      }
      return {LLVMBasicValType::i16, num_elts, true};
    case 2:
      if (num_elts % 4 == 0) {
        return {LLVMBasicValType::v4i32, num_elts / 4, num_elts != 4};
      } else if (num_elts % 2 == 0) {
        return {LLVMBasicValType::v2i32, num_elts / 2, num_elts != 2};
      }
      return {LLVMBasicValType::i32, num_elts, true};
    case 3:
      if (num_elts % 2 == 0) {
        return {LLVMBasicValType::v2i64, num_elts / 2, num_elts != 2};
      }
      return {LLVMBasicValType::i64, num_elts, true};
    default:
      // iN with N<8 or N>64.
      return {LLVMBasicValType::invalid, 0, false};
    }
  }
  case llvm::Type::FloatTyID:
    if (num_elts % 4 == 0) {
      return {LLVMBasicValType::v4f32, num_elts / 4, num_elts != 4};
    } else if (num_elts % 2 == 0) {
      return {LLVMBasicValType::v2f32, num_elts / 2, num_elts != 2};
    }
    return {LLVMBasicValType::f32, num_elts, true};
  case llvm::Type::DoubleTyID:
    if (num_elts % 2 == 0) {
      return {LLVMBasicValType::v2f64, num_elts / 2, num_elts != 2};
    }
    return {LLVMBasicValType::f64, num_elts, true};
  case llvm::Type::PointerTyID:
    if (el_ty->getPointerAddressSpace() == 0) {
      // TODO: for n=2, this should be {v2i64, 1, false}?
      return {LLVMBasicValType::i64, num_elts, true};
    }
    return {LLVMBasicValType::invalid, 0, false};
  default: return {LLVMBasicValType::invalid, 0, false};
  }
}

LLVMBasicValType
    LLVMAdaptor::lower_simple_type(const llvm::Type *type) noexcept {
  constexpr size_t LutBeginInt = llvm::Type::TargetExtTyID + 1;
  static constexpr auto type_lut = []() {
    std::array<LLVMBasicValType, LutBeginInt + 129> tys{};
    tys.fill(LLVMBasicValType::invalid);
    tys[llvm::Type::FloatTyID] = LLVMBasicValType::f32;
    tys[llvm::Type::DoubleTyID] = LLVMBasicValType::f64;
    tys[llvm::Type::FP128TyID] = LLVMBasicValType::f128;
    tys[llvm::Type::VoidTyID] = LLVMBasicValType::none;
    // TODO: check address space
    tys[llvm::Type::PointerTyID] = LLVMBasicValType::ptr;

    // Integer types
    for (unsigned i = 1; i <= 64; ++i) {
      tys[LutBeginInt + i] = i <= 8    ? LLVMBasicValType::i8
                             : i <= 16 ? LLVMBasicValType::i16
                             : i <= 32 ? LLVMBasicValType::i32
                                       : LLVMBasicValType::i64;
    }
    tys[LutBeginInt + 128] = LLVMBasicValType::i128;
    return tys;
  }();

  unsigned id = type->getTypeID();
  if (id == llvm::Type::IntegerTyID) {
    if (unsigned width = type->getIntegerBitWidth(); width <= 128) {
      id = LutBeginInt + width;
    }
  }
  return type_lut[id];
}

std::pair<unsigned, unsigned>
    LLVMAdaptor::complex_types_append(llvm::Type *type,
                                      size_t desc_idx) noexcept {
  auto ty = lower_simple_type(type);
  unsigned num;
  if (ty != LLVMBasicValType::invalid) {
    num = basic_ty_part_count(ty);
  } else if (auto fvt = llvm::dyn_cast<llvm::FixedVectorType>(type)) {
    bool scalarized = false;
    std::tie(ty, num, scalarized) = lower_vector_type(fvt);
    complex_part_types[desc_idx].desc.incompatible_layout |= scalarized;
  }

  if (ty != LLVMBasicValType::invalid) {
    unsigned size = basic_ty_part_size(ty);
    unsigned align = size * basic_ty_part_count(ty); // essentially just i128.
    assert(num > 0);
    // TODO: support types with different part types/sizes?
    for (unsigned i = 0; i < num; i++) {
      complex_part_types.emplace_back(ty, size, i == num - 1);
    }
    return std::make_pair(num * size, align);
  }

  size_t start = complex_part_types.size();
  switch (type->getTypeID()) {
  case llvm::Type::ArrayTyID: {
    auto [sz, algn] =
        complex_types_append(type->getArrayElementType(), desc_idx);
    size_t len = complex_part_types.size() - start;

    unsigned nelem = type->getArrayNumElements();
    complex_part_types.resize(start + nelem * len);
    for (unsigned i = 1; i < nelem; i++) {
      std::memcpy(&complex_part_types[start + i * len],
                  &complex_part_types[start],
                  len * sizeof(LLVMComplexPart));
    }
    if (nelem > 0) {
      if (nelem * len > LLVMComplexPart::MaxLength) {
        complex_part_types[desc_idx].desc.invalid = true;
      }
      complex_part_types[start].part.nest_inc++;
      complex_part_types[start + nelem * len - 1].part.nest_dec++;
    }
    return std::make_pair(nelem * sz, algn);
  }
  case llvm::Type::StructTyID: {
    unsigned size = 0;
    unsigned align = 1;
    bool packed = llvm::cast<llvm::StructType>(type)->isPacked();
    for (auto *el : llvm::cast<llvm::StructType>(type)->elements()) {
      unsigned prev = complex_part_types.size() - 1;
      auto [el_size, el_align] = complex_types_append(el, desc_idx);
      assert(el_size % el_align == 0 && "size must be multiple of alignment");
      if (packed) {
        el_align = 1;
      }

      unsigned old_size = size;
      size = tpde::util::align_up(size, el_align);
      if (size != old_size) {
        complex_part_types[prev].part.pad_after += size - old_size;
      }
      size += el_size;
      align = std::max(align, el_align);
    }

    size_t end = complex_part_types.size() - 1;
    unsigned old_size = size;
    size = tpde::util::align_up(size, align);
    if (size > 0) {
      if (end - start + 1 > LLVMComplexPart::MaxLength) {
        complex_part_types[desc_idx].desc.invalid = true;
      }
      complex_part_types[start].part.nest_inc++;
      complex_part_types[end].part.nest_dec++;
      complex_part_types[end].part.pad_after += size - old_size;
    }
    return std::make_pair(size, align);
  }
  default:
    complex_part_types[desc_idx].desc.invalid = true;
    return std::make_pair(0, 1);
  }
}

[[gnu::noinline]] std::pair<LLVMBasicValType, unsigned long>
    LLVMAdaptor::lower_complex_type(llvm::Type *type) noexcept {
  auto [it, inserted] = complex_type_map.try_emplace(type);
  if (inserted) {
    // Vector types are rather uncommon and non-trivial to handle, so handle
    // them here and cache the result. Not all vectors are actually complex.
    if (auto fvt = llvm::dyn_cast<llvm::FixedVectorType>(type)) {
      auto [ty, num, scalarized] = lower_vector_type(fvt);
      if (ty != LLVMBasicValType::invalid && !scalarized) {
        assert(num == 1 && "multi-part vector with compatible layout?");
        it->second = std::make_pair(ty, num);
        return it->second;
      }
    }

    unsigned start = complex_part_types.size();
    complex_part_types.push_back(LLVMComplexPart{}); // Complex part desc
    // TODO: store size/alignment?
    complex_types_append(type, start);
    unsigned len = complex_part_types.size() - (start + 1);
    complex_part_types[start].desc.num_parts = len;

    if (complex_part_types[start].desc.invalid) [[unlikely]] {
      report_unsupported_type(type);
      // Never store unsupported types in complex_type_map. The map lives
      // throughout the entire module, but func_unsupported (set by
      // report_unsupported_type) is reset for every function.
      complex_type_map.erase(it);
      return std::make_pair(LLVMBasicValType::invalid, ~0u);
    }

    it->second = std::make_pair(LLVMBasicValType::complex, start);
  }

  return it->second;
}

std::pair<unsigned, unsigned>
    LLVMAdaptor::complex_part_for_index(IRValueRef value,
                                        llvm::ArrayRef<unsigned> search) {
  ValueParts parts = val_parts(value);
  assert(parts.bvt == LLVMBasicValType::complex && parts.complex);
  unsigned part_count = parts.count();

  assert(search.size() > 0);

  unsigned depth = 0;
  tpde::util::SmallVector<unsigned, 16> indices;
  unsigned first_part = -1u;
  for (unsigned i = 0; i < part_count; i++) {
    indices.resize(indices.size() + parts.complex[i + 1].part.nest_inc);
    while (first_part == -1u && indices[depth] == search[depth]) {
      if (depth + 1 < search.size()) {
        depth++;
      } else {
        first_part = i;
      }
    }

    indices.resize(indices.size() - parts.complex[i + 1].part.nest_dec);
    if (parts.complex[i + 1].part.ends_value && !indices.empty()) {
      indices.back()++;
    }

    if (first_part != -1u &&
        (indices.size() <= depth || indices[depth] > search[depth])) {
      return std::make_pair(first_part, i);
    }
  }

  assert(0 && "out-of-range part index?");
  return std::make_pair(0, 0);
}

} // end namespace tpde_llvm
