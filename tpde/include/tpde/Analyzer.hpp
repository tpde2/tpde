// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include <algorithm>
#include <format>
#include <ostream>

#include "IRAdaptor.hpp"
#include "tpde/ValLocalIdx.hpp"
#include "tpde/base.hpp"
#include "util/SmallBitSet.hpp"
#include "util/SmallVector.hpp"

namespace tpde {

template <IRAdaptor Adaptor>
struct Analyzer {
  // some forwards for the IR type defs
  using IRValueRef = typename Adaptor::IRValueRef;
  using IRInstRef = typename Adaptor::IRInstRef;
  using IRBlockRef = typename Adaptor::IRBlockRef;
  using IRFuncRef = typename Adaptor::IRFuncRef;

  static constexpr IRBlockRef INVALID_BLOCK_REF = Adaptor::INVALID_BLOCK_REF;


  static constexpr size_t SMALL_BLOCK_NUM = 64;
  static constexpr size_t SMALL_VALUE_NUM = 128;

  /// Reference to the adaptor
  Adaptor *adaptor;

  /// An index into block_layout
  enum class BlockIndex : u32 {
  };
  static constexpr BlockIndex INVALID_BLOCK_IDX = static_cast<BlockIndex>(~0u);

  /// The block layout, a BlockIndex is an index into this array
  util::SmallVector<IRBlockRef, SMALL_BLOCK_NUM> block_layout = {};

  /// For each BlockIndex, the corresponding loop
  // TODO(ts): add the delayed free list in here to save on allocations?
  util::SmallVector<u32, SMALL_BLOCK_NUM> block_loop_map = {};

  struct Loop {
    u32 level;
    u32 parent;
    // [begin, end[
    BlockIndex begin = INVALID_BLOCK_IDX, end = INVALID_BLOCK_IDX;

    // for building the loop tree, we accumulate the number of blocks here
    u32 num_blocks = 0;
    // TODO(ts): add skip_target?

    u32 definitions = 0, definitions_in_childs = 0;
  };

  util::SmallVector<Loop, 16> loops = {};

  // TODO(ts): move all struct definitions to the top?
  struct LivenessInfo {
    // [first, last]
    BlockIndex first, last;
    u32 ref_count;
    u32 lowest_common_loop;

    // TODO(ts): maybe outsource both these booleans to a bitset?
    // we're wasting a lot of space here

    /// The value may not be deallocated until the last block is finished
    /// even if the reference count hits 0
    bool last_full;

    u16 epoch = 0;
  };

  util::SmallVector<LivenessInfo, SMALL_VALUE_NUM> liveness = {};
  /// Epoch of liveness information, entries with a value not equal to this
  /// epoch are invalid. This is an optimization to avoid clearing the entire
  /// liveness vector for every function, which is important for functions with
  /// many values that are ignored for the liveness analysis (e.g., var refs).
  u16 liveness_epoch = 0;
  u32 liveness_max_value;

  u32 num_insts;

  explicit Analyzer(Adaptor *adaptor) : adaptor(adaptor) {}

  /// Start the compilation of a new function and build the loop tree and
  /// liveness information. Previous information is discarded.
  void switch_func(IRFuncRef func);

  IRBlockRef block_ref(const BlockIndex idx) const noexcept {
    assert(static_cast<u32>(idx) <= block_layout.size());
    if (static_cast<u32>(idx) == block_layout.size()) {
      // this might be called with next_block() which is invalid for the
      // last block
      return INVALID_BLOCK_REF;
    }
    return block_layout[static_cast<u32>(idx)];
  }

  BlockIndex block_idx(IRBlockRef block_ref) const noexcept {
    return static_cast<BlockIndex>(adaptor->block_info(block_ref));
  }

  const LivenessInfo &liveness_info(const ValLocalIdx val_idx) const noexcept {
    assert(static_cast<u32>(val_idx) < liveness.size());
    assert(liveness[static_cast<u32>(val_idx)].epoch == liveness_epoch &&
           "access to liveness of ignored value");
    return liveness[static_cast<u32>(val_idx)];
  }

  u32 block_loop_idx(const BlockIndex idx) const noexcept {
    return block_loop_map[static_cast<u32>(idx)];
  }

  const Loop &loop_from_idx(const u32 idx) const noexcept { return loops[idx]; }

  bool block_has_multiple_incoming(const BlockIndex idx) const noexcept {
    return block_has_multiple_incoming(block_ref(idx));
  }

  bool block_has_multiple_incoming(IRBlockRef block_ref) const noexcept {
    return (adaptor->block_info2(block_ref) & 0b11) == 2;
  }

  bool block_has_phis(BlockIndex idx) const noexcept {
    return block_has_phis(block_ref(idx));
  }

  bool block_has_phis(IRBlockRef block_ref) const noexcept {
    return (adaptor->block_info2(block_ref) & 0b1'0000) != 0;
  }

  void print_rpo(std::ostream &os) const;
  void print_block_layout(std::ostream &os) const;
  void print_loops(std::ostream &os) const;
  void print_liveness(std::ostream &os) const;

protected:
  // for use during liveness analysis
  LivenessInfo &liveness_maybe(const IRValueRef val) noexcept;

  void build_block_layout();

  void build_loop_tree_and_block_layout(
      const util::SmallVector<IRBlockRef, SMALL_BLOCK_NUM> &block_rpo,
      const util::SmallVector<u32, SMALL_BLOCK_NUM> &loop_parent,
      const util::SmallBitSet<256> &loop_heads);

  /// Builds a vector of block references in reverse post-order.
  void build_rpo_block_order(
      util::SmallVector<IRBlockRef, SMALL_BLOCK_NUM> &out) const noexcept;

  /// Creates a bitset of loop_heads and sets the parent loop of each block.
  /// The u32 in loop_parent is an index into block_rpo
  void identify_loops(
      const util::SmallVector<IRBlockRef, SMALL_BLOCK_NUM> &block_rpo,
      util::SmallVector<u32, SMALL_BLOCK_NUM> &loop_parent,
      util::SmallBitSet<256> &loop_heads) const noexcept;

  void compute_liveness() noexcept;
};

template <IRAdaptor Adaptor>
void Analyzer<Adaptor>::switch_func([[maybe_unused]] IRFuncRef func) {
  build_block_layout();
  compute_liveness();
}

template <IRAdaptor Adaptor>
void Analyzer<Adaptor>::print_rpo(std::ostream &os) const {
  // build_rpo_block_order clobbers block data, so save and restore.
  util::SmallVector<std::tuple<IRBlockRef, u32, u32>, SMALL_BLOCK_NUM> data;
  for (IRBlockRef cur : adaptor->cur_blocks()) {
    data.emplace_back(cur, adaptor->block_info(cur), adaptor->block_info2(cur));
  }

  util::SmallVector<IRBlockRef, SMALL_BLOCK_NUM> rpo;
  build_rpo_block_order(rpo);
  for (u32 i = 0; i < rpo.size(); ++i) {
    os << std::format("  {}: {}\n", i, adaptor->block_fmt_ref(rpo[i]));
  }

  for (const auto &[cur, val1, val2] : data) {
    adaptor->block_set_info(cur, val1);
    adaptor->block_set_info2(cur, val2);
  }
}

template <IRAdaptor Adaptor>
void Analyzer<Adaptor>::print_block_layout(std::ostream &os) const {
  for (u32 i = 0; i < block_layout.size(); ++i) {
    os << std::format("  {}: {}\n", i, adaptor->block_fmt_ref(block_layout[i]));
  }
}

template <IRAdaptor Adaptor>
void Analyzer<Adaptor>::print_loops(std::ostream &os) const {
  for (u32 i = 0; i < loops.size(); ++i) {
    const auto &loop = loops[i];
    os << std::format("  {}: level {}, parent {}, {}->{}\n",
                      i,
                      loop.level,
                      loop.parent,
                      static_cast<u32>(loop.begin),
                      static_cast<u32>(loop.end));
  }
}

template <IRAdaptor Adaptor>
void Analyzer<Adaptor>::print_liveness(std::ostream &os) const {
  for (u32 i = 0; i <= liveness_max_value; ++i) {
    if (liveness[i].epoch != liveness_epoch) {
      os << std::format("  {}: ignored\n", i);
      continue;
    }

    const auto &info = liveness[i];
    os << std::format("  {}: {} refs, {}->{} ({}->{}), lf: {}\n",
                      i,
                      info.ref_count,
                      static_cast<u32>(info.first),
                      static_cast<u32>(info.last),
                      adaptor->block_fmt_ref(block_ref(info.first)),
                      adaptor->block_fmt_ref(block_ref(info.last)),
                      info.last_full);
  }
}

template <IRAdaptor Adaptor>
typename Analyzer<Adaptor>::LivenessInfo &
    Analyzer<Adaptor>::liveness_maybe(const IRValueRef val) noexcept {
  const ValLocalIdx val_idx = adaptor->val_local_idx(val);
  if constexpr (Adaptor::TPDE_PROVIDES_HIGHEST_VAL_IDX) {
    assert(liveness.size() > static_cast<u32>(val_idx));
    return liveness[static_cast<u32>(val_idx)];
  } else {
    if (liveness_max_value <= static_cast<u32>(val_idx)) {
      liveness_max_value = static_cast<u32>(val_idx);
      if (liveness.size() <= liveness_max_value) {
        // TODO: better growth strategy?
        liveness.resize(liveness_max_value + 0x100);
      }
    }
    return liveness[static_cast<u32>(val_idx)];
  }
}

template <IRAdaptor Adaptor>
void Analyzer<Adaptor>::build_block_layout() {
  util::SmallVector<IRBlockRef, SMALL_BLOCK_NUM> block_rpo{};
  build_rpo_block_order(block_rpo);

  util::SmallVector<u32, SMALL_BLOCK_NUM> loop_parent{};
  util::SmallBitSet<256> loop_heads{};

  // TODO(ts): print out this step?
  identify_loops(block_rpo, loop_parent, loop_heads);
  assert(loop_parent.size() == block_rpo.size());
  // the entry block is always the loop head for the root loop
  loop_heads.mark_set(0);

  build_loop_tree_and_block_layout(block_rpo, loop_parent, loop_heads);
}

template <IRAdaptor Adaptor>
void Analyzer<Adaptor>::build_loop_tree_and_block_layout(
    const util::SmallVector<IRBlockRef, SMALL_BLOCK_NUM> &block_rpo,
    const util::SmallVector<u32, SMALL_BLOCK_NUM> &loop_parent,
    const util::SmallBitSet<256> &loop_heads) {
  // TODO(ts): maybe merge this into the block_rpo?
  struct BlockLoopInfo {
    u32 loop_idx;
    u32 rpo_idx;
  };

  util::SmallVector<BlockLoopInfo, SMALL_BLOCK_NUM> loop_blocks;
  loop_blocks.resize_uninitialized(block_rpo.size());
  for (u32 i = 0; i < block_rpo.size(); ++i) {
    loop_blocks[i] = BlockLoopInfo{~0u, i};
  }

  // if we have not seen the parent loop before, we need to recursively insert
  // them.
  // the recursive call only happens with irreducible control-flow
  const auto build_or_get_parent_loop = [&](const u32 i,
                                            const auto &self) -> u32 {
    const auto parent = loop_parent[i];
    if (loop_blocks[parent].loop_idx != ~0u) {
      // we have already seen that block and given it a loop
      return loop_blocks[parent].loop_idx;
    } else {
      // get the parent loop and build the loop for this block
      const auto parent_loop_idx = self(parent, self);
      const auto loop_idx = loops.size();
      loops.push_back(Loop{.level = loops[parent_loop_idx].level + 1,
                           .parent = parent_loop_idx});
      loop_blocks[parent].loop_idx = loop_idx;
      return loop_idx;
    }
  };

  // entry is always the head of the top-level loop
  loops.clear();
  loops.push_back(Loop{.level = 0, .parent = 0, .num_blocks = 1});
  loop_blocks[0].loop_idx = 0;

  for (u32 i = 1; i < loop_parent.size(); ++i) {
    const u32 parent_loop =
        build_or_get_parent_loop(i, build_or_get_parent_loop);

    if (loop_heads.is_set(i)) {
      // if the loop is irreducible, we might have already inserted it so
      // check for that.
      //
      // NOTE: we could also get away with unsetting loop_heads for that
      // loop if it is irreducible if we would not count the loop_head in
      // its own loop's num_blocks but in its parent
      auto loop_idx = loop_blocks[i].loop_idx;
      if (loop_idx == ~0u) [[likely]] {
        loop_idx = loops.size();
        loops.push_back(
            Loop{.level = loops[parent_loop].level + 1, .parent = parent_loop});
        loop_blocks[i].loop_idx = loop_idx;
      }
      ++loops[loop_idx].num_blocks;
    } else {
      loop_blocks[i].loop_idx = parent_loop;
      ++loops[parent_loop].num_blocks;
    }
  }

  // acummulate the total number of blocks in a loop by iterating over them in
  // reverse-order. this works since we always push the parents first
  // leave out the first loop since it's its own parent
  for (u32 i = loops.size() - 1; i > 0; --i) {
    const auto &loop = loops[i];
    loops[loop.parent].num_blocks += loop.num_blocks;
  }

  assert(loops[0].num_blocks == block_rpo.size());

  // now layout the blocks by iterating in RPO and either place them at the
  // current offset of the parent loop or, if they are a new loop, place the
  // whole loop at the offset of the parent. this will ensure that blocks
  // surrounded by loops will also be layouted that way in the final order.
  //
  // however, the way this is implemented causes the loop index to not
  // correspond 1:1 with the final layout, though they will still be tightly
  // packed and only the order inside a loop may change. note(ts): this could
  // be mitigated with another pass i think.
  //
  // NB: we don't clear block_layout/block_loop_map, all entries are overwritten
  block_layout.resize_uninitialized(block_rpo.size());
  // TODO(ts): merge this with block_layout for less malloc calls?
  // however, we will mostly need this in the liveness computation so it may
  // be better for cache utilization to keep them separate
  block_loop_map.resize_uninitialized(block_rpo.size());

  loops[0].begin = loops[0].end = static_cast<BlockIndex>(0);

  const auto layout_loop = [&](const u32 loop_idx, const auto &self) -> void {
    assert(loops[loop_idx].begin == INVALID_BLOCK_IDX);
    const auto parent = loops[loop_idx].parent;
    if (loops[parent].begin == INVALID_BLOCK_IDX) {
      // should only happen with irreducible control-flow
      self(parent, self);
    }

    const auto loop_begin = loops[parent].end;
    loops[parent].end = static_cast<BlockIndex>(static_cast<u32>(loop_begin) +
                                                loops[loop_idx].num_blocks);
    assert(static_cast<u32>(loops[parent].end) -
               static_cast<u32>(loops[parent].begin) <=
           loops[parent].num_blocks);

    loops[loop_idx].begin = loops[loop_idx].end = loop_begin;
  };

  for (u32 i = 0u; i < block_rpo.size(); ++i) {
    const auto loop_idx = loop_blocks[i].loop_idx;
    if (loops[loop_idx].begin == INVALID_BLOCK_IDX) {
      layout_loop(loop_idx, layout_loop);
    }

    const auto block_ref = block_rpo[loop_blocks[i].rpo_idx];
    const auto block_idx = static_cast<u32>(loops[loop_idx].end);
    loops[loop_idx].end = static_cast<BlockIndex>(block_idx + 1);

    block_layout[block_idx] = block_ref;
    block_loop_map[block_idx] = loop_idx;
    adaptor->block_set_info(block_ref, block_idx);
  }

  assert(static_cast<u32>(loops[0].end) == block_rpo.size());
}

template <IRAdaptor Adaptor>
void Analyzer<Adaptor>::build_rpo_block_order(
    util::SmallVector<IRBlockRef, SMALL_BLOCK_NUM> &out) const noexcept {
  out.clear();

  u32 num_blocks = 0;
  {
    // Initialize the block info
    u32 idx = 0;
    for (IRBlockRef cur : adaptor->cur_blocks()) {
      adaptor->block_set_info(cur, idx);
      adaptor->block_set_info2(cur, 0);
      ++idx;
    }
    num_blocks = idx;
  }
  out.resize_uninitialized(num_blocks);

  // implement the RPO generation using a simple stack that also walks in
  // post-order and then reverse at the end. However, consider the following
  // CFG
  //
  // A:
  //  - B:
  //    - D
  //    - E
  //  - C:
  //    - F
  //    - G
  //
  // which has valid RPOs A B D E C F G, A B D E C G F, A B E D C F G, ...
  // which is not very nice for loops since in many IRs the order of the block
  // has some meaning for the layout so we sort the pushed children by the
  // order in the block list
  util::SmallVector<IRBlockRef, SMALL_BLOCK_NUM / 2> stack;
  stack.push_back(adaptor->cur_entry_block());

  // NOTE(ts): because we process children in reverse order
  // this gives a bit funky results with irreducible loops
  // but it should be fine I think
  auto rpo_idx = num_blocks - 1;
  while (!stack.empty()) {
    const auto cur_node = stack.back();

    // have we already added the block to the RPO list?
    // if so we just skip it
    if (adaptor->block_info2(cur_node) & 0b1000) {
      stack.pop_back();
      continue;
    }

    // have we already pushed the children of the nodes to the stack and
    // processed them? if so we can add the block to the post-order,
    // otherwise add the children and wait for them to be processed
    if (adaptor->block_info2(cur_node) & 0b100) {
      stack.pop_back();
      // set the blocks RPO index and push it to the RPO list
      adaptor->block_set_info(cur_node, rpo_idx);
      // mark the block as already being on the RPO list
      adaptor->block_set_info2(cur_node,
                               adaptor->block_info2(cur_node) | 0b1000);
      out[rpo_idx] = cur_node;
      --rpo_idx;
      continue;
    }

    // mark as visited and add the successors
    adaptor->block_set_info2(cur_node, adaptor->block_info2(cur_node) | 0b100);

    const auto start_idx = stack.size();
    // push the successors onto the stack to visit them
    for (const auto succ : adaptor->block_succs(cur_node)) {
      assert(succ != adaptor->cur_entry_block());
      const u32 info = adaptor->block_info2(succ);
      if ((info & 0b11) != 0) {
        if ((info & 0b11) == 1) {
          // note that succ has more than one incoming edge
          adaptor->block_set_info2(succ, (info & ~0b11) | 2);
        }
      } else {
        // TODO(ts): if we just do a post-order traversal and reverse
        // everything at the end we could use only block_info to check
        // whether we already saw a block. And I forgot why?
        adaptor->block_set_info2(succ, info | 1);
      }

      // if the successor is already on the rpo list or it has been
      // already visited and children added
      if (adaptor->block_info2(succ) & 0b1100) {
        continue;
      }

      stack.push_back(succ);
    }

    // Order the pushed children by their original block
    // index since the children get visited in reverse and then inserted
    // in reverse order in the rpo list @_@
    const auto len = stack.size() - start_idx;
    if (len <= 1) {
      continue;
    }

    if (len == 2) {
      if (adaptor->block_info(stack[start_idx]) >
          adaptor->block_info(stack[start_idx + 1])) {
        std::swap(stack[start_idx], stack[start_idx + 1]);
      }
      continue;
    }

    std::sort(stack.begin() + start_idx,
              stack.end(),
              [this](const IRBlockRef lhs, const IRBlockRef rhs) {
                // note(ts): this may have not so nice performance
                // characteristics if the block lookup is a hashmap so
                // maybe cache this for larger lists?
                return adaptor->block_info(lhs) < adaptor->block_info(rhs);
              });
  }

  if (rpo_idx != 0xFFFF'FFFF) {
    // there are unreachable blocks
    // so we did not fill up the whole array and need to shift it
    // TODO(ts): benchmark this against filling up a vector and always
    // reversing it tho it should be better i think
    out.erase(out.begin(), out.begin() + 1 + rpo_idx);

    // need to fixup the RPO index for blocks as well :/
    for (auto i = 0u; i < out.size(); ++i) {
      adaptor->block_set_info(out[i], i);
    }

    if constexpr (WithAsserts) {
      // In debug builds, reset block index of unreachable blocks.
      for (IRBlockRef cur : adaptor->cur_blocks()) {
        if (adaptor->block_info2(cur) == 0) {
          adaptor->block_set_info(cur, 0xFFFF'FFFF);
        }
      }
    }
  }

#ifdef TPDE_LOGGING
  TPDE_LOG_TRACE("Finished building RPO for blocks:");
  for (u32 i = 0; i < out.size(); ++i) {
    TPDE_LOG_TRACE("Index {}: {}", i, adaptor->block_fmt_ref(out[i]));
  }
#endif
}

template <IRAdaptor Adaptor>
void Analyzer<Adaptor>::identify_loops(
    const util::SmallVector<IRBlockRef, SMALL_BLOCK_NUM> &block_rpo,
    util::SmallVector<u32, SMALL_BLOCK_NUM> &loop_parent,
    util::SmallBitSet<256> &loop_heads) const noexcept {
  loop_parent.clear();
  loop_parent.resize(block_rpo.size());
  loop_heads.clear();
  loop_heads.resize(block_rpo.size());

  // Implement the modified algorithm from Wei et al.: A New Algorithm for
  // Identifying Loops in Decompilation
  // in a non-recursive form

  // TODO(ts): ask the adaptor if it can store this information for us?
  // then we could save the allocation here
  struct BlockInfo {
    bool traversed;
    bool self_loop;
    u32 dfsp_pos;
    u32 iloop_header;
  };

  util::SmallVector<BlockInfo, SMALL_BLOCK_NUM> block_infos;
  block_infos.resize(block_rpo.size());

  struct TravState {
    u32 block_idx; // b0 from the paper
    u32 dfsp_pos;
    u32 nh;
    u8 state = 0;
    decltype(adaptor->block_succs(adaptor->cur_entry_block()).begin()) succ_it;
    decltype(adaptor->block_succs(adaptor->cur_entry_block()).end()) end_it;
  };

  // The algorithm will reach the depth of the CFG so the small vector needs
  // to be relatively big
  // TODO(ts); maybe use the recursive form for small CFGs since that may be
  // faster or use stack switching since the non-recursive version is really
  // ugly
  util::SmallVector<TravState, SMALL_BLOCK_NUM> trav_state;

  trav_state.push_back(
      TravState{.block_idx = 0,
                .dfsp_pos = 1,
                .succ_it = adaptor->block_succs(block_rpo[0]).begin(),
                .end_it = adaptor->block_succs(block_rpo[0]).end()});

  const auto tag_lhead = [&block_infos](u32 b, u32 h) {
    if (b == h || h == 0) {
      return;
    }

    auto cur1 = b, cur2 = h;
    while (block_infos[cur1].iloop_header != 0) {
      const auto ih = block_infos[cur1].iloop_header;
      if (ih == cur2) {
        return;
      }
      if (block_infos[ih].dfsp_pos < block_infos[cur2].dfsp_pos) {
        block_infos[cur1].iloop_header = cur2;
        cur1 = cur2;
        cur2 = ih;
      } else {
        cur1 = ih;
      }
    }
    block_infos[cur1].iloop_header = cur2;
  };

  // TODO(ts): this can be optimized and more condensed so that the state
  // variable becomes unnecessary
  while (!trav_state.empty()) {
    auto &state = trav_state.back();
    const auto block_idx = state.block_idx;
    switch (state.state) {
    case 0: {
      // entry
      block_infos[block_idx].traversed = true;
      block_infos[block_idx].dfsp_pos = state.dfsp_pos;

      // TODO(ts): somehow make sure that the iterators can live when
      // succs is destroyed?
      // auto succs    = adaptor->block_succs(block_rpo[block_idx]);
      // state.succ_it = succs.begin();
      // state.end_it  = succs.end();
      state.state = 1;
    }
      [[fallthrough]];
    case 1: {
    loop_inner:
      auto cont = false;
      while (state.succ_it != state.end_it) {
        const auto succ_idx = adaptor->block_info(*state.succ_it);
        if (succ_idx == block_idx) {
          block_infos[block_idx].self_loop = true;
        }

        if (block_infos[succ_idx].traversed) {
          if (block_infos[succ_idx].dfsp_pos > 0) {
            tag_lhead(block_idx, succ_idx);
          } else if (block_infos[succ_idx].iloop_header != 0) {
            auto h_idx = block_infos[succ_idx].iloop_header;
            if (block_infos[h_idx].dfsp_pos > 0) {
              tag_lhead(block_idx, h_idx);
            } else {
              while (block_infos[h_idx].iloop_header != 0) {
                h_idx = block_infos[h_idx].iloop_header;
                if (block_infos[h_idx].dfsp_pos > 0) {
                  tag_lhead(block_idx, h_idx);
                  break;
                }
              }
            }
          }
          ++state.succ_it;
          continue;
        }

        // recurse
        cont = true;
        state.state = 2;
        // TODO(ts): somehow make sure that the iterators can live when
        // succs is destroyed?
        auto succs = adaptor->block_succs(block_rpo[succ_idx]);
        trav_state.push_back(TravState{.block_idx = succ_idx,
                                       .dfsp_pos = state.dfsp_pos + 1,
                                       .succ_it = succs.begin(),
                                       .end_it = succs.end()});
        break;
      }

      if (cont) {
        continue;
      }

      block_infos[block_idx].dfsp_pos = 0;
      trav_state.pop_back();
      if (trav_state.empty()) {
        break;
      }
      trav_state.back().nh = block_infos[block_idx].iloop_header;
      continue;
    }
    case 2: {
      const auto nh = state.nh;
      tag_lhead(block_idx, nh);

      ++state.succ_it;
      state.state = 1;
      goto loop_inner;
    }
    }
  }

  for (u32 i = 0; i < block_rpo.size(); ++i) {
    auto &info = block_infos[i];
    if (info.iloop_header != 0) {
      loop_parent[i] = info.iloop_header;
      loop_heads.mark_set(info.iloop_header);
    }
    if (info.self_loop) {
      loop_heads.mark_set(i);
    }
  }
}

template <IRAdaptor Adaptor>
void Analyzer<Adaptor>::compute_liveness() noexcept {
  // implement the liveness algorithm described in
  // http://databasearchitects.blogspot.com/2020/04/linear-time-liveness-analysis.html
  // and Kohn et al.: Adaptive Execution of Compiled Queries
  // TODO(ts): also expose the two-pass liveness algo as an option

  TPDE_LOG_TRACE("Starting Liveness Analysis");

  // Bump epoch. On overflow, we must clear all liveness info entries.
  if (++liveness_epoch == 0) {
    liveness.clear();
    liveness_epoch = 1;
  }

  if constexpr (Adaptor::TPDE_PROVIDES_HIGHEST_VAL_IDX) {
    liveness_max_value = adaptor->cur_highest_val_idx();
    if (liveness_max_value >= liveness.size()) {
      liveness.resize(liveness_max_value + 1);
    }
  } else {
    liveness_max_value = 0;
  }

  num_insts = 0;

  const auto visit = [this](const IRValueRef value, const u32 block_idx) {
    TPDE_LOG_TRACE("  Visiting value {} in block {}",
                   adaptor->value_fmt_ref(value),
                   block_idx);
    if (adaptor->val_ignore_in_liveness_analysis(value)) {
      TPDE_LOG_TRACE("    value is ignored");
      return;
    }

    auto &liveness = liveness_maybe(value);
    if (liveness.epoch != liveness_epoch) {
      TPDE_LOG_TRACE("    initializing liveness info, lcl is {}",
                     block_loop_map[block_idx]);
      liveness = LivenessInfo{
          .first = static_cast<BlockIndex>(block_idx),
          .last = static_cast<BlockIndex>(block_idx),
          .ref_count = 1,
          .lowest_common_loop = block_loop_map[block_idx],
          .last_full = false,
          .epoch = liveness_epoch,
      };
      return;
    }

    assert(liveness.ref_count != ~0u && "used value without definition");

    ++liveness.ref_count;
    TPDE_LOG_TRACE("    increasing ref_count to {}", liveness.ref_count);

    // helpers
    const auto update_for_block_only =
        [&liveness, block_idx](bool check_for_backwards_extension) {
          const auto old_first = static_cast<u32>(liveness.first);
          const auto old_last = static_cast<u32>(liveness.last);

          const auto new_first = std::min(old_first, block_idx);
          const auto new_last = std::max(old_last, block_idx);
          liveness.first = static_cast<BlockIndex>(new_first);
          liveness.last = static_cast<BlockIndex>(new_last);

          // if last changed, we don't need to extend the lifetime to the end
          // of the last block except if we extended the liveness interval
          // to an outside loop which means we saw a use in a previous block
          if (old_last == new_last && check_for_backwards_extension) {
            liveness.last_full = true;
          } else if (old_last != new_last) {
            liveness.last_full = false;
          }
        };

    const auto update_for_loop = [&liveness](
                                     const Loop &loop,
                                     bool check_for_backwards_extension) {
      const auto old_first = static_cast<u32>(liveness.first);
      const auto old_last = static_cast<u32>(liveness.last);

      const auto new_first = std::min(old_first, static_cast<u32>(loop.begin));
      const auto new_last = std::max(old_last, static_cast<u32>(loop.end) - 1);
      liveness.first = static_cast<BlockIndex>(new_first);
      liveness.last = static_cast<BlockIndex>(new_last);

      // if last changed, set last_full to true
      // since the values need to be allocated when the loop is active
      // Otherwise, if the liveness interval was extended backwards we need to
      // mark the liveness intervall as full as well
      if (old_last != new_last || check_for_backwards_extension) {
        liveness.last_full = true;
      }
    };

    const auto block_loop_idx = block_loop_map[block_idx];
    if (liveness.lowest_common_loop == block_loop_idx) {
      // just extend the liveness interval
      TPDE_LOG_TRACE("    lcl is same as block loop");
      update_for_block_only(false);

      TPDE_LOG_TRACE("    new interval {}->{}, lf: {}",
                     static_cast<u32>(liveness.first),
                     static_cast<u32>(liveness.last),
                     liveness.last_full);
      return;
    }

    const Loop &liveness_loop = loops[liveness.lowest_common_loop];
    const Loop &block_loop = loops[block_loop_idx];

    if (liveness_loop.level < block_loop.level &&
        static_cast<u32>(block_loop.begin) <
            static_cast<u32>(liveness_loop.end)) {
      assert(static_cast<u32>(block_loop.end) <=
             static_cast<u32>(liveness_loop.end));

      TPDE_LOG_TRACE("    block_loop {} is nested inside lcl", block_loop_idx);
      // The current use is nested inside the loop of the liveness
      // interval so we only need to get the loop at level
      // (liveness_loop.level + 1) that contains block_loop and extend the
      // liveness interval
      const auto target_level = liveness_loop.level + 1;
      auto cur_loop_idx = block_loop_idx;
      auto cur_level = block_loop.level;
      // TODO(ts): we could also skip some loops here since we know that
      // there have to be at least n=(cur_level-target_level) loops
      // between cur_loop and target_loop
      // so we could choose cur_loop_idx = min(cur_parent, cur_idx-n)?
      // however this might jump into more nested loops
      // so maybe check (cur_idx-n).level first?
      while (cur_level != target_level) {
        cur_loop_idx = loops[cur_loop_idx].parent;
        --cur_level;
      }
      assert(loops[cur_loop_idx].level == target_level);
      TPDE_LOG_TRACE("    target_loop is {}", cur_loop_idx);
      update_for_loop(loops[cur_loop_idx], false);

      TPDE_LOG_TRACE("    new interval {}->{}, lf: {}",
                     static_cast<u32>(liveness.first),
                     static_cast<u32>(liveness.last),
                     liveness.last_full);
      return;
    }

    TPDE_LOG_TRACE("    block_loop {} is nested higher or in a different loop",
                   block_loop_idx);
    // need to update the lowest common loop to contain both liveness_loop
    // and block_loop and then extend the interval accordingly

    // TODO(ts): this algorithm is currently worst-case O(n) which makes the
    // whole liveness analysis worst-case O(n^2) which is not good. However,
    // we expect the number of loops to be much smaller than the number of
    // values so it should be fine for most programs. To be safe, we should
    // implement the algorithm from "Gusfield: Constant-Time Lowest Common
    // Ancestor Retrieval" which seems to be the one with the most
    // reasonable overhead. It is however not zero and the queries are also
    // not exactly free so we should definitely benchmark both version first
    // with both small and large programs

    auto lhs_idx = liveness.lowest_common_loop;
    auto rhs_idx = block_loop_idx;
    auto prev_rhs = rhs_idx;
    auto prev_lhs = lhs_idx;
    while (lhs_idx != rhs_idx) {
      const auto lhs_level = loops[lhs_idx].level;
      const auto rhs_level = loops[rhs_idx].level;
      if (lhs_level > rhs_level) {
        prev_lhs = lhs_idx;
        lhs_idx = loops[lhs_idx].parent;
      } else if (lhs_level < rhs_level) {
        prev_rhs = rhs_idx;
        rhs_idx = loops[rhs_idx].parent;
      } else {
        prev_lhs = lhs_idx;
        prev_rhs = rhs_idx;
        lhs_idx = loops[lhs_idx].parent;
        rhs_idx = loops[rhs_idx].parent;
      }
    }

    assert(static_cast<u32>(loops[lhs_idx].begin) <=
           static_cast<u32>(liveness_loop.begin));
    assert(static_cast<u32>(loops[lhs_idx].end) >=
           static_cast<u32>(liveness_loop.end));
    TPDE_LOG_TRACE("    new lcl is {}", lhs_idx);

    liveness.lowest_common_loop = lhs_idx;

    // extend for the full loop that contains liveness_loop and is nested
    // directly in lcl
    assert(loops[prev_lhs].parent == lhs_idx);
    assert(static_cast<u32>(loops[prev_lhs].begin) <=
           static_cast<u32>(liveness_loop.begin));
    assert(static_cast<u32>(loops[prev_lhs].end) >=
           static_cast<u32>(liveness_loop.end));
    update_for_loop(loops[prev_lhs], false);

    // extend by block if the block_loop is the lcl
    // or by prev_rhs (the loop containing block_loop nested directly in
    // lcl) otherwise
    if (lhs_idx == block_loop_idx) {
      update_for_block_only(true);
    } else {
      assert(loops[prev_rhs].parent == lhs_idx);
      assert(loops[prev_rhs].level == loops[lhs_idx].level + 1);
      update_for_loop(loops[prev_rhs], true);
    }

    TPDE_LOG_TRACE("    new interval {}->{}, lf: {}",
                   static_cast<u32>(liveness.first),
                   static_cast<u32>(liveness.last),
                   liveness.last_full);
  };

  assert(block_layout[0] == adaptor->cur_entry_block());
  if constexpr (Adaptor::TPDE_LIVENESS_VISIT_ARGS) {
    for (const IRValueRef arg : adaptor->cur_args()) {
      visit(arg, 0);
    }
  }

  for (u32 block_idx = 0; block_idx < block_layout.size(); ++block_idx) {
    IRBlockRef block = block_layout[block_idx];
    TPDE_LOG_TRACE(
        "Analyzing block {} ('{}')", block_idx, adaptor->block_fmt_ref(block));
    const auto block_loop_idx = block_loop_map[block_idx];

    bool has_phis = false;
    for (const IRValueRef phi : adaptor->block_phis(block)) {
      TPDE_LOG_TRACE("Analyzing phi {}", adaptor->value_fmt_ref(phi));
      has_phis = true;

      const auto phi_ref = adaptor->val_as_phi(phi);
      const u32 slot_count = phi_ref.incoming_count();
      for (u32 i = 0; i < slot_count; ++i) {
        const IRBlockRef incoming_block = phi_ref.incoming_block_for_slot(i);
        const IRValueRef incoming_value = phi_ref.incoming_val_for_slot(i);
        if (adaptor->block_info2(incoming_block) == 0) {
          TPDE_LOG_TRACE("ignoring phi input from unreachable pred ({})",
                         adaptor->block_fmt_ref(incoming_block));
          continue;
        }
        const auto incoming_block_idx = adaptor->block_info(incoming_block);

        // mark the incoming value as used in the incoming block
        visit(incoming_value, incoming_block_idx);
        // mark the PHI-value as used in the incoming block
        visit(phi, incoming_block_idx);
      }
    }

    if (has_phis) {
      adaptor->block_set_info2(block, adaptor->block_info2(block) | 0b1'0000);
    }

    for (const IRInstRef inst : adaptor->block_insts(block)) {
      TPDE_LOG_TRACE("Analyzing instruction {}", adaptor->inst_fmt_ref(inst));
      for (const IRValueRef res : adaptor->inst_results(inst)) {
        // mark the value as used in the current block
        visit(res, block_idx);
        ++loops[block_loop_idx].definitions;
      }

      for (const IRValueRef operand : adaptor->inst_operands(inst)) {
        visit(operand, block_idx);
      }

      num_insts += 1;
    }
  }

  // fill out the definitions_in_childs counters
  // (skip 0 since it has itself as a parent)
  for (u32 idx = loops.size() - 1; idx != 0; --idx) {
    auto &loop = loops[idx];
    loops[loop.parent].definitions_in_childs +=
        loop.definitions_in_childs + loop.definitions;
  }

  if constexpr (WithAsserts) {
    // reset the incorrect ref_counts in the liveness infos
    for (auto &entry : liveness) {
      if (entry.ref_count == ~0u) {
        entry.ref_count = 0;
      }
    }
  }

  TPDE_LOG_TRACE("Finished Liveness Analysis");
}

} // namespace tpde
