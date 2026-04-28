"""
Tests for the checkpoint-restore rollback flow.

Covers:
  - database.save_conversation / get_recent_conversation / delete_last_n_turns
  - in-memory pop_last_turn / pop_n_turns (isolated, no Discord connection needed)
  - combined scenario: DB + memory stay consistent after rollback and simulated restart

Run:
    python -m pytest test_checkpoint_rollback.py -v
  or:
    python test_checkpoint_rollback.py
"""

import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))

import database
import bot


# ── helpers ──────────────────────────────────────────────────────────────────

def _build_temp_history_db(path: str) -> None:
    """Create a minimal conversation_history table at *path*."""
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversation_history (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            channel_id TEXT NOT NULL,
            user_id    TEXT NOT NULL,
            user_name  TEXT NOT NULL,
            content    TEXT NOT NULL,
            role       TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def _row_count(path: str, channel_id: str) -> int:
    conn = sqlite3.connect(path)
    n = conn.execute(
        "SELECT COUNT(*) FROM conversation_history WHERE channel_id = ?",
        (channel_id,),
    ).fetchone()[0]
    conn.close()
    return n


def _seed_memory(channel_id: str, n_turns: int) -> None:
    """Populate bot.conversation_contexts[channel_id] with *n_turns* turn-pairs."""
    ctx = []
    for i in range(1, n_turns + 1):
        ctx.append({"role": "user",      "content": f"user msg {i}"})
        ctx.append({"role": "assistant",  "content": f"bot msg {i}"})
    bot.conversation_contexts[channel_id] = ctx


def _clear_memory(channel_id: str) -> None:
    bot.conversation_contexts.pop(channel_id, None)


# ── test cases ────────────────────────────────────────────────────────────────

class TestDeleteLastNTurns(unittest.TestCase):
    """Unit tests for database.delete_last_n_turns."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self._db_path = self._tmp.name
        _build_temp_history_db(self._db_path)
        self._orig_path = database.HISTORY_DB
        database.HISTORY_DB = self._db_path

    def tearDown(self):
        database.HISTORY_DB = self._orig_path
        os.unlink(self._db_path)

    def _save(self, channel_id: str, turns: int) -> None:
        """Insert *turns* complete turn-pairs (user + assistant) for channel."""
        for i in range(1, turns + 1):
            database.save_conversation(channel_id, "u1", "User", f"user msg {i}", "user")
            database.save_conversation(channel_id, "u1", "Bot",  f"bot  msg {i}", "assistant")

    def test_delete_exact_turns(self):
        ch = "ch-001"
        self._save(ch, 5)
        self.assertEqual(_row_count(self._db_path, ch), 10)

        database.delete_last_n_turns(ch, 2)

        self.assertEqual(_row_count(self._db_path, ch), 6)

    def test_delete_all_turns(self):
        ch = "ch-002"
        self._save(ch, 3)
        database.delete_last_n_turns(ch, 3)
        self.assertEqual(_row_count(self._db_path, ch), 0)

    def test_delete_more_than_exist_is_safe(self):
        """Requesting more turns than exist must not raise and must leave the
        table empty (or partially trimmed), not crash."""
        ch = "ch-003"
        self._save(ch, 2)
        database.delete_last_n_turns(ch, 100)
        self.assertEqual(_row_count(self._db_path, ch), 0)

    def test_delete_zero_turns_is_noop(self):
        ch = "ch-004"
        self._save(ch, 3)
        database.delete_last_n_turns(ch, 0)
        self.assertEqual(_row_count(self._db_path, ch), 6)

    def test_other_channels_unaffected(self):
        """Rollback on channel A must not touch channel B's rows."""
        ch_a, ch_b = "ch-005a", "ch-005b"
        self._save(ch_a, 4)
        self._save(ch_b, 3)
        database.delete_last_n_turns(ch_a, 2)
        self.assertEqual(_row_count(self._db_path, ch_b), 6)


class TestGetRecentConversationAfterRollback(unittest.TestCase):
    """Simulates a bot restart: after rollback, get_recent_conversation must
    return only the surviving rows — never the deleted ones."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self._db_path = self._tmp.name
        _build_temp_history_db(self._db_path)
        self._orig_path = database.HISTORY_DB
        database.HISTORY_DB = self._db_path

    def tearDown(self):
        database.HISTORY_DB = self._orig_path
        os.unlink(self._db_path)

    def test_deleted_turns_do_not_reappear_after_restart(self):
        ch = "ch-restart"
        # Insert 5 turns
        for i in range(1, 6):
            database.save_conversation(ch, "u1", "User", f"user msg {i}", "user")
            database.save_conversation(ch, "u1", "Bot",  f"bot msg {i}",  "assistant")

        # Roll back the last 2 turns (simulate checkpoint-restore)
        database.delete_last_n_turns(ch, 2)

        # Simulate restart: fetch history up to a generous limit
        history = database.get_recent_conversation(ch, limit=20)

        # Only 3 turns (6 rows) should survive
        self.assertEqual(len(history), 6)

        # The deleted content must not appear
        surviving_contents = {row["content"] for row in history}
        for deleted_i in (4, 5):
            self.assertNotIn(f"user msg {deleted_i}", surviving_contents)
            self.assertNotIn(f"bot msg {deleted_i}",  surviving_contents)

    def test_content_order_preserved_after_rollback(self):
        ch = "ch-order"
        for i in range(1, 4):
            database.save_conversation(ch, "u1", "User", f"user msg {i}", "user")
            database.save_conversation(ch, "u1", "Bot",  f"bot msg {i}",  "assistant")

        database.delete_last_n_turns(ch, 1)

        history = database.get_recent_conversation(ch, limit=20)
        contents = [row["content"] for row in history]
        self.assertEqual(contents, [
            "user msg 1", "bot msg 1",
            "user msg 2", "bot msg 2",
        ])


class TestInMemoryPopNTurns(unittest.TestCase):
    """Unit tests for bot.pop_last_turn / bot.pop_n_turns.

    Each test uses a unique channel id and tears down via _clear_memory so
    the shared conversation_contexts dict stays clean between tests.
    """

    CH = "ch-mem-"

    def _ch(self, suffix: str) -> str:
        return f"{self.CH}{suffix}"

    def tearDown(self):
        for suffix in ("a", "b", "c", "d", "e"):
            _clear_memory(self._ch(suffix))

    def test_pop_last_turn_removes_one_pair(self):
        ch = self._ch("a")
        _seed_memory(ch, 3)
        bot.pop_last_turn(ch)
        ctx = bot.conversation_contexts[ch]
        self.assertEqual(len(ctx), 4)
        self.assertEqual(ctx[-1]["content"], "bot msg 2")

    def test_pop_n_turns_removes_n_pairs(self):
        ch = self._ch("b")
        _seed_memory(ch, 5)
        bot.pop_n_turns(ch, 3)
        ctx = bot.conversation_contexts[ch]
        self.assertEqual(len(ctx), 4)
        self.assertEqual(ctx[-1]["content"], "bot msg 2")

    def test_pop_n_turns_zero_is_noop(self):
        ch = self._ch("c")
        _seed_memory(ch, 3)
        bot.pop_n_turns(ch, 0)
        ctx = bot.conversation_contexts[ch]
        self.assertEqual(len(ctx), 6)

    def test_pop_n_turns_more_than_exist_empties_context(self):
        ch = self._ch("d")
        _seed_memory(ch, 2)
        bot.pop_n_turns(ch, 10)
        ctx = bot.conversation_contexts.get(ch, [])
        self.assertEqual(ctx, [])

    def test_pop_last_turn_on_missing_channel_is_safe(self):
        ch = self._ch("e")
        _clear_memory(ch)
        bot.pop_last_turn(ch)  # must not raise


class TestCombinedRollback(unittest.TestCase):
    """End-to-end checkpoint-restore: DB trim + memory rollback must agree."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self._db_path = self._tmp.name
        _build_temp_history_db(self._db_path)
        self._orig_path = database.HISTORY_DB
        database.HISTORY_DB = self._db_path

    def test_memory_and_db_agree_after_rollback(self):
        ch = "ch-combined"
        n_total = 5
        n_rollback = 2

        # Seed bot.conversation_contexts and DB together
        _seed_memory(ch, n_total)
        for i in range(1, n_total + 1):
            database.save_conversation(ch, "u1", "User", f"user msg {i}", "user")
            database.save_conversation(ch, "u1", "Bot",  f"bot msg {i}",  "assistant")

        # Perform rollback via the real bot functions (mirrors ConfirmCheckpointView.confirm)
        bot.pop_n_turns(ch, n_rollback)
        database.delete_last_n_turns(ch, n_rollback)

        expected_turns = n_total - n_rollback
        ctx = bot.conversation_contexts.get(ch, [])

        # In-memory check
        self.assertEqual(len(ctx), expected_turns * 2)

        # DB check (simulated restart)
        history = database.get_recent_conversation(ch, limit=50)
        self.assertEqual(len(history), expected_turns * 2)

        # Both agree on the surviving content
        mem_contents = [m["content"] for m in ctx]
        db_contents  = [row["content"] for row in history]
        self.assertEqual(mem_contents, db_contents)

    def tearDown(self):
        _clear_memory("ch-combined")
        database.HISTORY_DB = self._orig_path
        os.unlink(self._db_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
