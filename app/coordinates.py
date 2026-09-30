"""批量偏移单位换算（纯函数，不依赖数据库）。

浏览器（JavaScript 字符串按 **UTF-16 码元**计偏移）与外部工具（常按
**UTF-8 字节**计偏移）提交的选区偏移，需要先换算成本服务统一使用的
**Unicode 码点**偏移，否则 emoji（如 ``🎉``，码点 1 个、UTF-16 两个码元、
UTF-8 四个字节）等字符会让选区错位。

本模块对目标版本全文**一次扫描**构建三种单位之间的边界映射，再对请求中的
每个半开区间做 O(1) 换算，整体复杂度 **O(n + m)**（n 为码点数、m 为区间数）：

- :func:`build_boundary_map`：单遍扫描全文，记录每个码点边界（字符间隙）
  在三种单位下的偏移，并标记落在 UTF-16 代理对 / UTF-8 多字节字符**内部**
  的非法位置；
- :func:`convert_intervals`：按请求声明的输入单位校验并逐项换算，越界、
  ``start > end`` 或端点落在字符内部时**整批拒绝**（不四舍五入、不猜测），
  一次性返回带请求下标与端点定位的全部错误。

合法偏移只能落在**字符边界**上：``[0, 文本末端]`` 之间的每个码点间隙。
``start == end`` 的空区间合法（选区可为空），只要两端点本身是合法边界。
相同输入必然得到完全一致的结果。
"""
from __future__ import annotations

from array import array
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Sequence, Tuple

# 支持的输入 / 输出单位
UNIT_CODEPOINT = "codepoint"
UNIT_UTF16 = "utf16"
UNIT_UTF8_BYTE = "utf8_byte"
UNITS: Tuple[str, ...] = (UNIT_CODEPOINT, UNIT_UTF16, UNIT_UTF8_BYTE)

Unit = Literal["codepoint", "utf16", "utf8_byte"]

# 区间数上限（与路由 schema 保持一致；纯函数不重复强制，仅作防御性常量）
MAX_INTERVALS = 200

# 槽位含义（用于 UTF-16 / UTF-8 的反向数组）：
# 槽值 >= 0 表示该偏移是字符边界，值为对应的码点边界下标；
# 槽值 == _SLOT_INTERIOR 表示该偏移落在某个多码元 / 多字节字符内部。
_SLOT_INTERIOR = -1


@dataclass(frozen=True)
class BoundaryMap:
    """目标版本全文在三种单位下的边界映射（一次扫描构建，复用 O(m) 次）。

    - ``codepoint_count`` / ``utf16_length`` / ``utf8_length``：三种单位下的
      文本长度（即末端边界偏移）；
    - ``codepoint_to_utf16[k]`` / ``codepoint_to_utf8[k]``：码点边界 ``k``
      （``0 <= k <= codepoint_count``）对应的 UTF-16 / UTF-8 偏移（正向）；
    - ``utf16_slots`` / ``utf8_slots``：长度为「对应单位长度 + 1」的反向数组，
      边界槽位存对应码点边界下标（>= 0），字符内部槽位为 :data:`_SLOT_INTERIOR`。
      正向、反向各 O(1)，故每个端点换算为常数时间。
    """

    codepoint_count: int
    utf16_length: int
    utf8_length: int
    codepoint_to_utf16: Tuple[int, ...]
    codepoint_to_utf8: Tuple[int, ...]
    utf16_slots: array
    utf8_slots: array

    def is_boundary(self, unit: Unit, offset: int) -> bool:
        """``offset`` 是否为 ``unit`` 单位下的合法字符边界（含文本末端）。"""
        if offset < 0:
            return False
        if unit == UNIT_CODEPOINT:
            return offset <= self.codepoint_count
        if unit == UNIT_UTF16:
            return (
                offset <= self.utf16_length
                and self.utf16_slots[offset] != _SLOT_INTERIOR
            )
        return (
            offset <= self.utf8_length
            and self.utf8_slots[offset] != _SLOT_INTERIOR
        )

    def to_codepoint(self, unit: Unit, offset: int) -> int:
        """把 ``unit`` 单位下的**合法边界**偏移换算为码点边界下标（O(1)）。

        调用方须先用 :meth:`is_boundary` 确认；字符内部位置不得调用本方法。
        """
        if unit == UNIT_CODEPOINT:
            return offset
        if unit == UNIT_UTF16:
            return self.utf16_slots[offset]
        return self.utf8_slots[offset]

    def from_codepoint(self, unit: Unit, cp: int) -> int:
        """把码点边界 ``cp`` 换算为 ``unit`` 单位下的偏移（O(1)）。"""
        if unit == UNIT_CODEPOINT:
            return cp
        if unit == UNIT_UTF16:
            return self.codepoint_to_utf16[cp]
        return self.codepoint_to_utf8[cp]


def _utf16_width(ch: str) -> int:
    """单个码点的 UTF-16 码元数：BMP 外（含 emoji）为代理对 2，其余 1。"""
    return 2 if ord(ch) > 0xFFFF else 1


def _utf8_width(ch: str) -> int:
    """单个码点的 UTF-8 字节数（1～4）。"""
    code = ord(ch)
    if code < 0x80:
        return 1
    if code < 0x800:
        return 2
    if code < 0x10000:
        return 3
    return 4


def build_boundary_map(text: str) -> BoundaryMap:
    """单遍扫描全文，构建三种单位的边界映射，时间与空间均为 O(n)。

    对每个码点累加其 UTF-16 / UTF-8 宽度：正向数组记录该码点之后的边界偏移；
    反向槽位数组先补「宽度 - 1」个字符内部槽（代理间隙 / 多字节内部），
    再补一个值为「码点边界下标」的边界槽。
    """
    cp_to_u16: List[int] = [0]
    cp_to_u8: List[int] = [0]
    u16_slots: array = array("q", [0])
    u8_slots: array = array("q", [0])

    u16_pos = 0
    u8_pos = 0
    for cp_index, ch in enumerate(text):
        w16 = _utf16_width(ch)
        w8 = _utf8_width(ch)
        u16_pos += w16
        u8_pos += w8
        cp_to_u16.append(u16_pos)
        cp_to_u8.append(u8_pos)
        # 宽度 - 1 个字符内部槽（BMP 内单码元 / ASCII 单字节时为 0 个）
        for _ in range(w16 - 1):
            u16_slots.append(_SLOT_INTERIOR)
        for _ in range(w8 - 1):
            u8_slots.append(_SLOT_INTERIOR)
        # 该码点之后的边界槽，值为码点边界下标 cp_index + 1
        u16_slots.append(cp_index + 1)
        u8_slots.append(cp_index + 1)

    return BoundaryMap(
        codepoint_count=len(text),
        utf16_length=u16_pos,
        utf8_length=u8_pos,
        codepoint_to_utf16=tuple(cp_to_u16),
        codepoint_to_utf8=tuple(cp_to_u8),
        utf16_slots=u16_slots,
        utf8_slots=u8_slots,
    )


@dataclass(frozen=True)
class CoordinateError:
    """一条区间校验错误（带请求下标与端点定位）。

    - ``error``：机器可读错误码；
    - ``message``：中文说明（含单位、偏移与文本长度）；
    - ``path``：请求 JSON 路径，如 ``intervals[3].start``；
    - ``index``：区间在请求中的下标；
    - ``endpoint``：``start`` / ``end`` / ``range``（整体顺序错误时）。
    """

    error: str
    message: str
    path: str
    index: int
    endpoint: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": self.error,
            "message": self.message,
            "path": self.path,
            "index": self.index,
            "endpoint": self.endpoint,
        }


def _interior_error(
    unit: Unit, index: int, endpoint: str, offset: int
) -> CoordinateError:
    if unit == UNIT_UTF16:
        what = "一个 UTF-16 代理对（高/低代理项之间）"
        code = "utf16_inside_surrogate_pair"
    else:
        what = "一个 UTF-8 多字节字符的内部"
        code = "utf8_inside_multibyte_char"
    return CoordinateError(
        error=code,
        message=(
            f"intervals[{index}].{endpoint}={offset} 落在{what}，不是合法字符边界；"
            "偏移不得取整或猜测，请改用字符边界"
        ),
        path=f"intervals[{index}].{endpoint}",
        index=index,
        endpoint=endpoint,
    )


def _out_of_bounds_error(
    unit: Unit, index: int, endpoint: str, offset: int, length: int
) -> CoordinateError:
    return CoordinateError(
        error="offset_out_of_bounds",
        message=(
            f"intervals[{index}].{endpoint}={offset} 超出文本范围："
            f"按 {unit} 计文本长度为 {length}，合法边界为 [0, {length}]"
        ),
        path=f"intervals[{index}].{endpoint}",
        index=index,
        endpoint=endpoint,
    )


def _check_endpoint(
    bmap: BoundaryMap, unit: Unit, index: int, endpoint: str, offset: int
) -> List[CoordinateError]:
    """校验单个端点：先越界、再字符内部；合法时无错误。"""
    if not bmap.is_boundary(unit, offset):
        if unit == UNIT_CODEPOINT:
            length = bmap.codepoint_count
            return [_out_of_bounds_error(unit, index, endpoint, offset, length)]
        length = bmap.utf16_length if unit == UNIT_UTF16 else bmap.utf8_length
        if offset < 0 or offset > length:
            return [_out_of_bounds_error(unit, index, endpoint, offset, length)]
        # 在范围内但不是边界 → 落在字符内部
        return [_interior_error(unit, index, endpoint, offset)]
    return []


def convert_intervals(
    bmap: BoundaryMap, unit: Unit, intervals: Sequence[Tuple[int, int]]
) -> Tuple[List[Dict[str, Any]], List[CoordinateError]]:
    """按声明单位批量校验并换算半开区间。

    返回 ``(results, errors)``：

    - 全部合法时 ``errors`` 为空，``results`` 逐项给出三种单位的对应范围与
      原索引（``index`` 即请求下标）；
    - 任一区间非法时 ``results`` 为空，``errors`` 一次性列出全部错误，
      按 ``(index, 端点顺序 start→end→range)`` 升序，**整批不换算、不猜测**。

    每个区间先独立校验 start / end（越界、字符内部），两端点均合法后再比较
    顺序；``start == end`` 的空区间合法。映射一次构建、各端点 O(1) 换算，
    整体 O(n + m)。
    """
    errors: List[CoordinateError] = []
    converted: List[Dict[str, Any]] = []

    for index, (raw_start, raw_end) in enumerate(intervals):
        ep_errors = _check_endpoint(bmap, unit, index, "start", raw_start)
        ep_errors += _check_endpoint(bmap, unit, index, "end", raw_end)
        if ep_errors:
            errors.extend(ep_errors)
            continue

        # 两端点均为合法边界：换算到码点并比较顺序
        cp_start = bmap.to_codepoint(unit, raw_start)
        cp_end = bmap.to_codepoint(unit, raw_end)
        if cp_start > cp_end:
            errors.append(
                CoordinateError(
                    error="start_after_end",
                    message=(
                        f"intervals[{index}] 的 start={raw_start} 大于 end={raw_end}"
                        f"（按 {unit} 计），半开区间要求 start <= end"
                    ),
                    path=f"intervals[{index}]",
                    index=index,
                    endpoint="range",
                )
            )
            continue

        converted.append(
            {
                "index": index,
                "codepoint": {"start": cp_start, "end": cp_end},
                "utf16": {
                    "start": bmap.from_codepoint(UNIT_UTF16, cp_start),
                    "end": bmap.from_codepoint(UNIT_UTF16, cp_end),
                },
                "utf8_byte": {
                    "start": bmap.from_codepoint(UNIT_UTF8_BYTE, cp_start),
                    "end": bmap.from_codepoint(UNIT_UTF8_BYTE, cp_end),
                },
            }
        )

    if errors:
        # 稳定排序：按下标，其次按 start(0) → end(1) → range(2)
        rank = {"start": 0, "end": 1, "range": 2}
        errors.sort(key=lambda e: (e.index, rank[e.endpoint]))
        return [], errors

    return converted, []
