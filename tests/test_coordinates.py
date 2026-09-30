"""批量偏移单位换算测试：边界映射纯函数 + 端到端接口。

覆盖：
- emoji（BMP 外，UTF-16 代理对 / UTF-8 四字节）、组合字符（UTF-8 多字节）、
  中文（BMP 内三字节、UTF-16 单码元）三类字符的换算；
- 一次构建映射、逐项 O(1) 换算（O(n+m)）与原索引回显；
- 越界、start>end、落在 UTF-16 代理对 / UTF-8 多字节字符内部时整批 422，
  错误带条目下标与端点定位、一次性列全、稳定排序；
- 空文本、空区间、空批次；只读与同一快照字节稳定。
"""
import json

import pytest
from fastapi.testclient import TestClient

from app.coordinates import (
    UNITS,
    build_boundary_map,
    convert_intervals,
)
from app.main import app


# ---------------------------------------------------------------------------
# 纯函数：边界映射
# ---------------------------------------------------------------------------


def test_boundary_lengths_for_emoji_combining_chinese():
    # a(1/1/1) 🎉(cp1/u16 2/u8 4) b(1/1/1) 中(1/1/3) + ◌́(cp1/u16 1/u8 2)
    # 一个文本同时含 emoji、中文与组合字符（组合符附在「中」上）
    text = "a🎉b中́"
    bm = build_boundary_map(text)
    assert bm.codepoint_count == 5
    assert bm.utf16_length == 6
    assert bm.utf8_length == 11
    assert bm.codepoint_to_utf16 == (0, 1, 3, 4, 5, 6)
    assert bm.codepoint_to_utf8 == (0, 1, 5, 6, 9, 11)


def test_forward_and_reverse_lookups_round_trip():
    text = "🎉中a𩸽，é🎉"
    bm = build_boundary_map(text)
    for cp in range(bm.codepoint_count + 1):
        u16 = bm.from_codepoint("utf16", cp)
        u8 = bm.from_codepoint("utf8_byte", cp)
        assert bm.to_codepoint("utf16", u16) == cp
        assert bm.to_codepoint("utf8_byte", u8) == cp
        assert bm.is_boundary("utf16", u16)
        assert bm.is_boundary("utf8_byte", u8)


def test_interior_slots_match_width_gap():
    text = "🎉中🎉"
    bm = build_boundary_map(text)
    # cp=3, u16=5(两个代理对各+1), u8=11(4+3+4)
    assert bm.utf16_length == 5
    assert bm.utf8_length == 11
    u16_interior = [i for i, v in enumerate(bm.utf16_slots) if v == -1]
    u8_interior = [i for i, v in enumerate(bm.utf8_slots) if v == -1]
    assert u16_interior == [1, 4]  # 两个 emoji 的代理间隙
    # emoji[0,4) 内部 1,2,3；中[4,7) 内部 5,6；emoji[7,11) 内部 8,9,10
    assert u8_interior == [1, 2, 3, 5, 6, 8, 9, 10]


def test_boundary_map_handles_long_text_without_overflow():
    # 码点下标远超 255：反向数组必须是宽整型数组而非 bytearray
    text = "🎉中" * 5000
    bm = build_boundary_map(text)
    assert bm.codepoint_count == 10000
    assert bm.utf16_length == 15000
    assert bm.utf16_slots[15000] == 10000


def test_empty_text_boundary_map():
    bm = build_boundary_map("")
    assert bm.codepoint_count == bm.utf16_length == bm.utf8_length == 0
    assert list(bm.utf16_slots) == [0]
    assert list(bm.utf8_slots) == [0]


# ---------------------------------------------------------------------------
# 纯函数：换算与三类整批拒绝
# ---------------------------------------------------------------------------


def test_convert_emoji_interval_all_units():
    bm = build_boundary_map("a🎉b中")  # cp4 / u16 5 / u8 9
    # emoji 在码点 [1,2)、UTF-16 [1,3)、UTF-8 [1,5)
    for unit, start, end in (
        ("codepoint", 1, 2),
        ("utf16", 1, 3),
        ("utf8_byte", 1, 5),
    ):
        res, errs = convert_intervals(bm, unit, [(start, end)])
        assert errs == []
        item = res[0]
        assert item["index"] == 0
        assert item["codepoint"] == {"start": 1, "end": 2}
        assert item["utf16"] == {"start": 1, "end": 3}
        assert item["utf8_byte"] == {"start": 1, "end": 5}


def test_convert_chinese_bmp_is_single_utf16_code_unit():
    bm = build_boundary_map("a中")  # cp2 / u16 2 / u8 4
    res, errs = convert_intervals(bm, "codepoint", [(1, 2)])
    assert errs == []
    assert res[0]["utf16"] == {"start": 1, "end": 2}  # 中文 UTF-16 仅 1 码元
    assert res[0]["utf8_byte"] == {"start": 1, "end": 4}  # 但 UTF-8 占 3 字节


def test_convert_combining_character_boundary():
    # 用 chr(0x301) 显式拼出组合重音，避免源码被规范化成预组合字符 U+00E9。
    text = "e" + chr(0x301) + "x"  # cp3 / u16 3 / u8 1+2+1=4
    assert [hex(ord(c)) for c in text] == ["0x65", "0x301", "0x78"]
    bm = build_boundary_map(text)
    assert bm.codepoint_count == 3 and bm.utf16_length == 3 and bm.utf8_length == 4
    # u8: e[0,1) 组合符[1,3)（内部偏移 2 非法） x[3,4)
    assert bm.is_boundary("utf8_byte", 1)
    assert not bm.is_boundary("utf8_byte", 2)  # 组合符内部
    assert bm.is_boundary("utf8_byte", 3)
    res, errs = convert_intervals(bm, "utf8_byte", [(1, 3)])
    assert errs == []
    assert res[0]["codepoint"] == {"start": 1, "end": 2}  # 组合符本身
    # 落在组合符内部同样整批拒绝
    res, errs = convert_intervals(bm, "utf8_byte", [(1, 2)])
    assert errs and errs[0].error == "utf8_inside_multibyte_char"


def test_start_after_end_rejected():
    bm = build_boundary_map("abc🎉")
    for unit in UNITS:
        res, errs = convert_intervals(bm, unit, [(1, 0)])
        assert res == []
        assert len(errs) == 1
        assert errs[0].error == "start_after_end"
        assert errs[0].endpoint == "range"
        assert errs[0].path == "intervals[0]"


def test_empty_interval_is_valid_at_any_boundary():
    bm = build_boundary_map("a🎉")  # cp2 / u16 3 / u8 5
    for unit, end in (("codepoint", 2), ("utf16", 3), ("utf8_byte", 5)):
        res, errs = convert_intervals(bm, unit, [(end, end)])
        assert errs == []
        assert res[0]["codepoint"] == {"start": 2, "end": 2}


def test_out_of_bounds_rejected_for_all_units():
    bm = build_boundary_map("a🎉")  # cp2 / u16 3 / u8 5
    cases = {
        "codepoint": 3,
        "utf16": 4,
        "utf8_byte": 6,
    }
    for unit, bad in cases.items():
        res, errs = convert_intervals(bm, unit, [(0, bad)])
        assert res == []
        assert errs[0].error == "offset_out_of_bounds"
        assert errs[0].endpoint == "end"
        # 末端边界本身合法
        res, errs = convert_intervals(bm, unit, [(0, bad - 1)])
        assert errs == []

def test_utf16_surrogate_pair_interior_rejected():
    bm = build_boundary_map("a🎉")  # emoji 占 UTF-16 [1,3)，偏移 2 在代理对之间
    for endpoint, interval in (("start", (2, 3)), ("end", (1, 2))):
        res, errs = convert_intervals(bm, "utf16", [interval])
        assert res == []
        assert len(errs) == 1
        assert errs[0].error == "utf16_inside_surrogate_pair"
        assert errs[0].endpoint == endpoint
        assert errs[0].path == f"intervals[0].{endpoint}"


def test_utf8_multibyte_interior_rejected():
    bm = build_boundary_map("a🎉")  # emoji 占 UTF-8 [1,5)，2/3/4 在字符内部
    for interior in (2, 3, 4):
        res, errs = convert_intervals(bm, "utf8_byte", [(interior, 5)])
        assert res == []
        assert errs[0].error == "utf8_inside_multibyte_char"
        assert errs[0].endpoint == "start"


def test_batch_collects_all_errors_and_rejects_whole_batch():
    bm = build_boundary_map("a🎉b中")  # cp4 / u16 5 / u8 9
    # 0: 起点落在代理对内部；1: start>end；2: 终点越界；3: 合法
    intervals = [(2, 3), (4, 1), (0, 99), (0, 1)]
    res, errs = convert_intervals(bm, "utf16", intervals)
    assert res == []  # 整批不出结果
    by_index = {}
    for e in errs:
        by_index.setdefault(e.index, []).append((e.endpoint, e.error))
    assert by_index[0] == [("start", "utf16_inside_surrogate_pair")]
    assert by_index[1] == [("range", "start_after_end")]
    assert by_index[2] == [("end", "offset_out_of_bounds")]
    assert 3 not in by_index
    # 稳定排序：index 升序、端点 start(0)→end(1)→range(2)
    rank = {"start": 0, "end": 1, "range": 2}
    keys = [(e.index, rank[e.endpoint]) for e in errs]
    assert keys == sorted(keys)


def test_both_endpoints_interior_reported_separately():
    bm = build_boundary_map("🎉")  # u16 [0,2)，偏移 1 在代理对内部
    res, errs = convert_intervals(bm, "utf16", [(1, 1)])
    assert res == []
    assert [e.endpoint for e in errs] == ["start", "end"]
    assert all(e.error == "utf16_inside_surrogate_pair" for e in errs)


def test_interval_index_is_echoed_in_order():
    bm = build_boundary_map("abcde")
    res, errs = convert_intervals(
        bm, "codepoint", [(4, 5), (0, 1), (2, 3)]
    )
    assert errs == []
    assert [r["index"] for r in res] == [0, 1, 2]
    assert res[0]["codepoint"] == {"start": 4, "end": 5}


def test_empty_text_only_zero_zero_valid():
    bm = build_boundary_map("")
    for unit in UNITS:
        res, errs = convert_intervals(bm, unit, [(0, 0)])
        assert errs == [] and res[0]["codepoint"] == {"start": 0, "end": 0}
        res, errs = convert_intervals(bm, unit, [(0, 1)])
        assert errs and errs[0].error == "offset_out_of_bounds"


def test_error_to_dict_shape():
    bm = build_boundary_map("🎉")
    _, errs = convert_intervals(bm, "utf16", [(1, 2)])
    d = errs[0].to_dict()
    assert set(d) == {"error", "message", "path", "index", "endpoint"}
    assert d["path"] == "intervals[0].start"
    assert d["index"] == 0 and d["endpoint"] == "start"


# ---------------------------------------------------------------------------
# 端到端接口
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _make_doc(client, content, title="偏移换算测试"):
    r = client.post("/documents", json={"title": title, "content": content})
    assert r.status_code == 201, r.text
    return r.json()["document_id"]


def _convert(client, doc, number, unit, intervals):
    return client.post(
        f"/documents/{doc}/versions/{number}/coordinates",
        json={"unit": unit, "intervals": [
            {"start": s, "end": e} for s, e in intervals
        ]},
    )


def test_api_convert_emoji_from_utf16(client):
    text = "你好🎉世界"  # cp5 / u16 6 / u8 (3+3+4+3+3)=16
    doc = _make_doc(client, text)
    # 浏览器视角：🎉 位于 UTF-16 [2,4)
    r = _convert(client, doc, 1, "utf16", [(2, 4)])
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["document_id"] == doc
    assert data["version_number"] == 1
    assert data["unit"] == "utf16"
    assert data["source_length"] == {"codepoint": 5, "utf16": 6, "utf8_byte": 16}
    item = data["intervals"][0]
    assert item["index"] == 0
    assert item["codepoint"] == {"start": 2, "end": 3}
    assert item["utf16"] == {"start": 2, "end": 4}
    assert item["utf8_byte"] == {"start": 6, "end": 10}
    assert data["interval_count"] == 1


def test_api_convert_from_utf8_with_combining_and_chinese(client):
    # 用 chr(0x301) 显式拼出组合重音，避免源码被规范化成预组合字符 U+00E9。
    # cp5 / u16 6 / u8 11：u8 a[0,1) 中[1,4) 🎉[4,8) e[8,9) ◌́[9,11)
    text = "a中🎉" + "e" + chr(0x301)
    doc = _make_doc(client, text)
    # 外部工具按 UTF-8：中文 [1,4)、emoji [4,8)、组合重音 [9,11)
    r = _convert(client, doc, 1, "utf8_byte", [(1, 4), (4, 8), (9, 11)])
    assert r.status_code == 200, r.text
    intervals = r.json()["intervals"]
    assert [i["index"] for i in intervals] == [0, 1, 2]
    assert intervals[0]["codepoint"] == {"start": 1, "end": 2}  # 中
    assert intervals[1]["codepoint"] == {"start": 2, "end": 3}  # 🎉
    assert intervals[1]["utf16"] == {"start": 2, "end": 4}
    assert intervals[2]["codepoint"] == {"start": 4, "end": 5}  # ◌́


def test_api_batch_utf16_surrogate_interior_returns_422(client):
    text = "x🎉y"  # u16: x[0,1)，emoji[1,3)（偏移 2 在代理对之间），y[3,4)
    doc = _make_doc(client, text)
    r = _convert(client, doc, 1, "utf16", [(2, 3), (0, 1)])
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert detail["unit"] == "utf16"
    errors = detail["errors"]
    assert errors[0]["error"] == "utf16_inside_surrogate_pair"
    assert errors[0]["index"] == 0 and errors[0]["endpoint"] == "start"
    assert errors[0]["path"] == "intervals[0].start"
    # 整批拒绝：合法的第二项也不出现在结果中
    assert "intervals" not in detail


def test_api_utf8_multibyte_and_out_of_bounds_and_order(client):
    text = "🎉中"  # cp2 / u16 3 / u8 7：emoji[0,4) 中[4,7)
    doc = _make_doc(client, text)
    r = client.post(
        f"/documents/{doc}/versions/1/coordinates",
        json={"unit": "utf8_byte", "intervals": [
            {"start": 2, "end": 4},   # 0: 起点落在 emoji 内部（4 是边界）
            {"start": 7, "end": 4},   # 1: start>end（7、4 均为合法边界）
            {"start": 0, "end": 99},  # 2: 终点越界
        ]},
    )
    assert r.status_code == 422
    errors = r.json()["detail"]["errors"]
    by_index = {e["index"]: (e["endpoint"], e["error"]) for e in errors}
    assert by_index[0] == ("start", "utf8_inside_multibyte_char")
    assert by_index[1] == ("range", "start_after_end")
    assert by_index[2] == ("end", "offset_out_of_bounds")
    # 下标升序
    assert [e["index"] for e in errors] == sorted(e["index"] for e in errors)

    # 两端点都落在多字节字符内部时各自报告一条（按 start→end 排序）
    r2 = client.post(
        f"/documents/{doc}/versions/1/coordinates",
        json={"unit": "utf8_byte", "intervals": [
            {"start": 2, "end": 3},
        ]},
    )
    assert r2.status_code == 422
    errs2 = r2.json()["detail"]["errors"]
    assert [(e["endpoint"], e["error"]) for e in errs2] == [
        ("start", "utf8_inside_multibyte_char"),
        ("end", "utf8_inside_multibyte_char"),
    ]


def test_api_empty_interval_and_empty_batch(client):
    doc = _make_doc(client, "🎉")  # u8 长度 4
    # 空区间 [end,end) 落在末端边界合法
    r = _convert(client, doc, 1, "utf8_byte", [(4, 4)])
    assert r.status_code == 200
    assert r.json()["intervals"][0]["codepoint"] == {"start": 1, "end": 1}
    # 空批次合法，返回空列表
    r = client.post(
        f"/documents/{doc}/versions/1/coordinates",
        json={"unit": "codepoint", "intervals": []},
    )
    assert r.status_code == 200
    assert r.json()["intervals"] == []
    assert r.json()["interval_count"] == 0


def test_api_more_than_200_intervals_rejected_by_schema(client):
    doc = _make_doc(client, "abc")
    r = client.post(
        f"/documents/{doc}/versions/1/coordinates",
        json={"unit": "codepoint",
              "intervals": [{"start": 0, "end": 1}] * 201},
    )
    assert r.status_code == 422


def test_api_unknown_unit_rejected_by_schema(client):
    doc = _make_doc(client, "abc")
    r = client.post(
        f"/documents/{doc}/versions/1/coordinates",
        json={"unit": "byte", "intervals": [{"start": 0, "end": 1}]},
    )
    assert r.status_code == 422


def test_api_negative_offset_rejected_by_schema(client):
    doc = _make_doc(client, "abc")
    r = client.post(
        f"/documents/{doc}/versions/1/coordinates",
        json={"unit": "codepoint", "intervals": [{"start": -1, "end": 1}]},
    )
    assert r.status_code == 422


def test_api_document_or_version_missing_404(client):
    doc = _make_doc(client, "abc")
    assert _convert(client, doc, 99, "codepoint", [(0, 1)]).status_code == 404
    assert _convert(client, 999999, 1, "codepoint", [(0, 1)]).status_code == 404


def test_api_works_on_specific_version_snapshot(client):
    doc = _make_doc(client, "a🎉")
    r = client.post(f"/documents/{doc}/versions", json={"content": "ab🎉cd"})
    assert r.status_code == 201
    # 第 1 版 cp2/u16 3/u8 5；emoji 码点 [1,2)，UTF-16 [1,3)
    r1 = _convert(client, doc, 1, "utf16", [(1, 3)])
    assert r1.json()["intervals"][0]["codepoint"] == {"start": 1, "end": 2}
    # 第 2 版 cp5/u16 6/u8 8；emoji 码点 [2,3)，UTF-16 [2,4)
    r2 = _convert(client, doc, 2, "utf16", [(2, 4)])
    assert r2.status_code == 200
    assert r2.json()["intervals"][0]["codepoint"] == {"start": 2, "end": 3}


def test_api_read_only_and_byte_stable(client):
    text = "甲🎉乙🎉丙"
    doc = _make_doc(client, text)
    body = {
        "unit": "utf16",
        "intervals": [{"start": 1, "end": 3}, {"start": 4, "end": 6}],
    }
    r1 = client.post(f"/documents/{doc}/versions/1/coordinates", json=body)
    r2 = client.post(f"/documents/{doc}/versions/1/coordinates", json=body)
    assert r1.status_code == r2.status_code == 200
    b1 = json.dumps(r1.json(), ensure_ascii=False, sort_keys=True,
                    separators=(",", ":")).encode("utf-8")
    b2 = json.dumps(r2.json(), ensure_ascii=False, sort_keys=True,
                    separators=(",", ":")).encode("utf-8")
    assert b1 == b2
    # 只读：没有新增版本 / 批注
    assert client.get(f"/documents/{doc}").json()["annotation_count"] == 0
    assert len(client.get(f"/documents/{doc}/versions").json()["versions"]) == 1
