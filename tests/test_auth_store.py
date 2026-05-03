from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from vn_labor_law_ai_assistant.auth_store import AuthStore


class AuthStoreTest(TestCase):
    def test_login_session_and_user_scoped_conversation_history(self) -> None:
        with TemporaryDirectory() as tmpdir:
            store = AuthStore(Path(tmpdir) / "app.db")
            user = store.authenticate_user("user@example.com", "user12345")
            self.assertIsNotNone(user)
            assert user is not None

            token, _ = store.create_session(user)
            self.assertEqual(store.get_user_by_token(token), user)

            conversation = store.create_conversation(
                user_id=user.id,
                title="Tôi nghỉ việc phải báo trước bao lâu?",
            )
            store.append_message(
                conversation_id=conversation["id"],
                role="user",
                content="Tôi nghỉ việc phải báo trước bao lâu?",
            )
            store.append_message(
                conversation_id=conversation["id"],
                role="assistant",
                content="Cần kiểm tra Điều 35 Bộ luật Lao động 2019.",
            )

            conversations = store.list_conversations(user_id=user.id)
            self.assertEqual(len(conversations), 1)
            self.assertEqual(conversations[0]["message_count"], 2)

            messages = store.list_messages(
                user_id=user.id,
                conversation_id=conversation["id"],
            )
            self.assertIsNotNone(messages)
            self.assertEqual([message["role"] for message in messages or []], ["user", "assistant"])

            admin = store.authenticate_user("admin@example.com", "admin12345")
            self.assertIsNotNone(admin)
            assert admin is not None
            self.assertEqual(
                store.list_messages(
                    user_id=admin.id,
                    conversation_id=conversation["id"],
                ),
                None,
            )

            store.revoke_session(token)
            self.assertIsNone(store.get_user_by_token(token))
