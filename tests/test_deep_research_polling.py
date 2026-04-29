"""
Test kịch bản polling cho deep research API.

Chạy: python -m pytest tests/test_deep_research_polling.py -v -s
Hoặc: python tests/test_deep_research_polling.py

Yêu cầu: AI service đang chạy tại BASE_URL (mặc định http://localhost:8001)
"""

import time
import requests
import sys

BASE_URL = "http://localhost:8001/api"
POLL_INTERVAL = 3   # giây
MAX_WAIT = 300       # giây (5 phút)


def color(text, code): return f"\033[{code}m{text}\033[0m"
def green(t): return color(t, 92)
def red(t):   return color(t, 91)
def yellow(t): return color(t, 93)
def bold(t):  return color(t, 1)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def start_research(query: str) -> dict:
    """POST /research/deep-research → nhận task_id ngay lập tức."""
    t0 = time.time()
    resp = requests.post(
        f"{BASE_URL}/research/deep-research",
        json={"query": query},
        timeout=10,
    )
    elapsed = time.time() - t0
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert "task_id" in data, f"No task_id in response: {data}"
    assert data["status"] == "pending", f"Expected pending, got: {data['status']}"
    print(f"  → task_id: {data['task_id']}  (response time: {elapsed:.2f}s)")
    assert elapsed < 5, f"POST took too long: {elapsed:.2f}s (should be < 5s)"
    return data


def poll_until_done(task_id: str) -> dict:
    """Poll GET /research/status/{task_id} cho đến khi done hoặc timeout."""
    start = time.time()
    polls = 0
    while True:
        elapsed = time.time() - start
        if elapsed > MAX_WAIT:
            raise TimeoutError(f"Job {task_id} chưa xong sau {MAX_WAIT}s")

        time.sleep(POLL_INTERVAL)
        polls += 1
        resp = requests.get(f"{BASE_URL}/research/status/{task_id}", timeout=10)

        if resp.status_code == 404:
            raise AssertionError(f"Task not found: {task_id}")
        if resp.status_code == 500:
            raise AssertionError(f"Server error: {resp.json().get('detail')}")

        assert resp.status_code == 200, f"Unexpected status {resp.status_code}"
        data = resp.json()
        status = data.get("status")
        print(f"  poll #{polls} ({elapsed:.0f}s): {status}")

        if status == "done":
            print(f"  → Hoàn thành sau {elapsed:.1f}s, {polls} lần poll")
            return data


def validate_result(data: dict):
    """Kiểm tra cấu trúc kết quả trả về."""
    assert data["status"] == "done"
    assert isinstance(data.get("answer"), str) and len(data["answer"]) > 0, "answer rỗng"
    assert isinstance(data.get("sources"), list), "sources phải là list"
    assert isinstance(data.get("confidence_score"), float), "confidence_score phải là float"
    assert isinstance(data.get("iterations_used"), int), "iterations_used phải là int"
    assert "planner_decision" in data, "thiếu planner_decision"
    print(f"  → answer length: {len(data['answer'])} chars")
    print(f"  → sources: {len(data['sources'])}")
    print(f"  → confidence: {data['confidence_score']:.2f}")
    print(f"  → iterations: {data['iterations_used']}")
    print(f"  → decision: {data.get('decision')}")


# ─────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────

def test_service_is_up():
    """TC1: Service có đang chạy không."""
    print(bold("\nTC1: Kiểm tra service health"))
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        # 200 hoặc 404 đều OK — miễn là service phản hồi
        print(f"  → HTTP {resp.status_code} — service đang chạy {green('✓')}")
    except requests.ConnectionError:
        print(red("  ✗ Không kết nối được. Hãy chạy: docker-compose up"))
        sys.exit(1)


def test_post_returns_task_id_immediately():
    """TC2: POST phải trả về task_id trong < 5 giây."""
    print(bold("\nTC2: POST trả về task_id ngay lập tức"))
    data = start_research("what is transformer architecture in deep learning")
    print(f"  {green('✓')} task_id nhận được ngay, không bị timeout")
    return data["task_id"]


def test_poll_returns_pending_then_done():
    """TC3: Full flow — start → poll → nhận kết quả đầy đủ."""
    print(bold("\nTC3: Full polling flow"))
    data = start_research("explain attention mechanism in neural networks")
    task_id = data["task_id"]

    result = poll_until_done(task_id)
    validate_result(result)
    print(f"  {green('✓')} Kết quả hợp lệ")


def test_invalid_task_id_returns_404():
    """TC4: task_id không tồn tại phải trả về 404."""
    print(bold("\nTC4: task_id không tồn tại → 404"))
    resp = requests.get(f"{BASE_URL}/research/status/invalid-uuid-xyz", timeout=5)
    assert resp.status_code == 404, f"Expected 404, got {resp.status_code}"
    print(f"  {green('✓')} 404 đúng như mong đợi")


def test_result_cleaned_up_after_fetch():
    """TC5: Sau khi lấy kết quả, task_id bị xóa → poll lần 2 phải 404."""
    print(bold("\nTC5: Task bị xóa sau khi lấy kết quả"))
    data = start_research("brief overview of reinforcement learning")
    task_id = data["task_id"]

    result = poll_until_done(task_id)
    assert result["status"] == "done"

    # Poll lần 2 — phải 404
    time.sleep(1)
    resp = requests.get(f"{BASE_URL}/research/status/{task_id}", timeout=5)
    assert resp.status_code == 404, f"Expected 404 after cleanup, got {resp.status_code}"
    print(f"  {green('✓')} Task đã được dọn sạch sau khi deliver")


def test_empty_query_handled():
    """TC6: Query rỗng — service không crash."""
    print(bold("\nTC6: Query rỗng"))
    try:
        resp = requests.post(
            f"{BASE_URL}/research/deep-research",
            json={"query": ""},
            timeout=10,
        )
        # 422 (validation error) hoặc 200 đều chấp nhận được
        print(f"  → HTTP {resp.status_code} — {green('✓')} service không crash")
    except Exception as e:
        print(f"  {red('✗')} Exception: {e}")
        raise


# ─────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print(bold("=" * 55))
    print(bold("  Deep Research Polling — Test Suite"))
    print(bold(f"  Target: {BASE_URL}"))
    print(bold("=" * 55))

    tests = [
        ("TC1", test_service_is_up),
        ("TC2", test_post_returns_task_id_immediately),
        ("TC4", test_invalid_task_id_returns_404),
        ("TC6", test_empty_query_handled),
        ("TC3", test_poll_returns_pending_then_done),   # chạy sau vì mất thời gian
        ("TC5", test_result_cleaned_up_after_fetch),    # chạy sau vì mất thời gian
    ]

    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except (AssertionError, TimeoutError, Exception) as e:
            print(f"  {red('✗ FAILED')}: {e}")
            failed += 1

    print(bold("\n" + "=" * 55))
    print(f"  Kết quả: {green(str(passed) + ' passed')}  {red(str(failed) + ' failed') if failed else ''}")
    print(bold("=" * 55))
    sys.exit(0 if failed == 0 else 1)
