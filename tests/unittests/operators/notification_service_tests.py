"""
Unit tests for the notification service.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import asyncio
import datetime
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from bson import Decimal128, ObjectId
import pytest
from pymongo.errors import OperationFailure

from fiftyone.operators.message import MessageData, MessageMetadata
from fiftyone.operators.store.notification_service import (
    MongoChangeStreamNotificationService,
)


class TestMongoChangeStreamNotificationService(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        self.remote_notifier = MagicMock()
        self.remote_notifier.broadcast_to_store = AsyncMock()

        self.collection = MagicMock()
        self.db = MagicMock()
        self.db.__getitem__.return_value = self.collection

        with patch("fiftyone.core.odm.get_async_db_conn") as mock_get_db:
            mock_get_db.return_value = self.db
            self.notification_service = MongoChangeStreamNotificationService(
                collection_name="test_collection",
                remote_notifier=self.remote_notifier,
            )
            self.notification_service._collection_async = self.collection

    def tearDown(self):
        self.loop.close()

    def test_notifications_serialize_nested_job_dates_without_mutation(self):
        requested_at = datetime.datetime(2026, 1, 1)
        job = {"progress": {"requested_at": requested_at}}
        message = MessageData(key="job", value=job, metadata=MessageMetadata())
        self.loop.run_until_complete(
            self.notification_service.notify("selection_jobs", message)
        )
        self.loop.run_until_complete(asyncio.sleep(0))
        store, payload = self.remote_notifier.broadcast_to_store.call_args.args
        self.assertEqual(store, "selection_jobs")
        value = json.loads(payload)["value"]["progress"]["requested_at"]
        self.assertEqual(value["_cls"], "DateTime")
        self.assertEqual(value["datetime"], 1767225600000)
        self.assertIs(job["progress"]["requested_at"], requested_at)

    def test_subscribe(self):
        callback = MagicMock()
        subscription_id = self.notification_service.subscribe(
            "test_store", callback, dataset_id="test_dataset"
        )

        self.assertIsNotNone(subscription_id)
        subscribers = (
            self.notification_service._subscription_registry.get_subscribers(
                "test_store"
            )
        )
        self.assertIn(subscription_id, subscribers)
        self.assertEqual(subscribers[subscription_id][0], callback)
        self.assertEqual(subscribers[subscription_id][1], "test_dataset")

    def test_unsubscribe(self):
        callback = MagicMock()
        subscription_id = self.notification_service.subscribe(
            "test_store", callback
        )

        self.notification_service.unsubscribe(subscription_id)

        subscribers = (
            self.notification_service._subscription_registry.get_subscribers(
                "test_store"
            )
        )
        self.assertNotIn(subscription_id, subscribers)

    def test_unsubscribe_all(self):
        callback1 = MagicMock()
        callback2 = MagicMock()

        self.notification_service.subscribe("test_store", callback1)
        self.notification_service.subscribe("test_store", callback2)

        self.notification_service.unsubscribe_all("test_store")

        subscribers = (
            self.notification_service._subscription_registry.get_subscribers(
                "test_store"
            )
        )
        self.assertEqual(len(subscribers), 0)

    def test_notify_with_dataset_id_filtering(self):
        # Subscribe with dataset_id filter
        callback1 = MagicMock()
        callback2 = MagicMock()
        self.notification_service.subscribe(
            "test_store", callback1, dataset_id="dataset1"
        )
        self.notification_service.subscribe(
            "test_store", callback2, dataset_id="dataset2"
        )

        # Notification for dataset1
        metadata = MessageMetadata(dataset_id="dataset1")
        message_data = MessageData(
            key="test_key", value="test_value", metadata=metadata
        )

        asyncio.run(
            self.notification_service.notify("test_store", message_data)
        )

        callback1.assert_called_once_with(message_data)
        callback2.assert_not_called()
        self.remote_notifier.broadcast_to_store.assert_called_once_with(
            "test_store", message_data.to_json()
        )

    def test_notify_without_dataset_id(self):
        # Subscribe without dataset_id filter
        callback = MagicMock()
        self.notification_service.subscribe("test_store", callback)

        # Notification without dataset_id
        metadata = MessageMetadata()
        message_data = MessageData(
            key="test_key", value="test_value", metadata=metadata
        )

        asyncio.run(
            self.notification_service.notify("test_store", message_data)
        )

        callback.assert_called_once_with(message_data)
        self.remote_notifier.broadcast_to_store.assert_called_once_with(
            "test_store", message_data.to_json()
        )


class TestMongoChangeStreamNotificationServiceAsync:
    @pytest.fixture
    def notification_service(self):
        remote_notifier = MagicMock()
        remote_notifier.broadcast_to_store = AsyncMock()

        collection = MagicMock()
        db = MagicMock()
        db.__getitem__.return_value = collection

        with patch("fiftyone.core.odm.get_async_db_conn") as mock_get_db:
            mock_get_db.return_value = db
            notification_service = MongoChangeStreamNotificationService(
                collection_name="test_collection",
                remote_notifier=remote_notifier,
            )
            # Initialize collection manually for testing
            notification_service._collection_async = collection
            yield notification_service, collection, remote_notifier

    @pytest.mark.asyncio
    async def test_handle_change_insert(self, notification_service):
        service, collection, remote_notifier = notification_service
        callback = MagicMock()
        service.subscribe("test_store", callback)

        change = {
            "operationType": "insert",
            "fullDocument": {
                "store_name": "test_store",
                "dataset_id": "test_dataset",
                "key": "test_key",
                "value": "test_value",
            },
            "wallTime": datetime.datetime.now(),
        }

        await service._handle_change(change)
        await asyncio.gather(*service._background_tasks)

        callback.assert_called_once()
        called_message = callback.call_args[0][0]

        assert called_message.key == "test_key"
        assert called_message.value == "test_value"
        assert called_message.metadata.dataset_id == "test_dataset"
        assert called_message.metadata.operation_type == "insert"

    @pytest.mark.asyncio
    async def test_handle_change_delete(self, notification_service):
        service, collection, remote_notifier = notification_service
        callback = MagicMock()
        service.subscribe("test_store", callback)

        # Mock a delete operation where fullDocument is not available
        change = {
            "operationType": "delete",
            "documentKey": {
                "_id": {
                    "store_name": "test_store",
                    "dataset_id": "test_dataset",
                    "key": "test_key",
                }
            },
            "wallTime": datetime.datetime.now(),
        }

        await service._handle_change(change)
        await asyncio.gather(*service._background_tasks)

        callback.assert_called_once()
        called_message = callback.call_args[0][0]

        assert called_message.key == "test_key"
        assert called_message.value is None
        assert called_message.metadata.dataset_id == "test_dataset"
        assert called_message.metadata.operation_type == "delete"

    @pytest.mark.asyncio
    async def test_run_with_change_stream(self, notification_service):
        service, collection, remote_notifier = notification_service

        with patch.object(
            service, "_run_change_stream"
        ) as mock_run_change_stream:
            mock_run_change_stream.return_value = None

            task = asyncio.create_task(service._run())

            await asyncio.sleep(0.1)

            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

            mock_run_change_stream.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_fallback_to_polling(self, notification_service):
        service, collection, remote_notifier = notification_service

        # Patch the _run method and check its calls
        with patch.object(
            service, "_run_change_stream"
        ) as mock_run_change_stream, patch.object(
            service, "_start_polling"
        ) as mock_start_polling:
            # Make _run_change_stream raise an exception to trigger fallback
            mock_run_change_stream.side_effect = OperationFailure(
                "Change stream not available"
            )

            # Make _start_polling just return a future that resolves immediately
            mock_start_polling.return_value = asyncio.Future()
            mock_start_polling.return_value.set_result(None)

            task = asyncio.create_task(service._run())

            await asyncio.sleep(0.1)

            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

            # Verify that fallback to polling occurred
            mock_run_change_stream.assert_called_once()
            mock_start_polling.assert_called_once()

    @pytest.mark.asyncio
    async def test_poll(self, notification_service):
        service, collection, remote_notifier = notification_service

        with patch.object(
            MongoChangeStreamNotificationService, "_get_current_stores"
        ) as mock_get_current_stores:
            # Setup
            mock_get_current_stores.return_value = ["test_store"]

            # Mock the collection's distinct method
            current_keys = {"key1", "key2", "key3"}
            collection.distinct = AsyncMock(return_value=current_keys)

            # Set up previous state
            service._last_poll_time = datetime.datetime.now(
                datetime.timezone.utc
            )
            service._last_keys = {"test_store": {"key1", "key2", "key4"}}

            # Mock find to return updated documents
            mock_cursor = MagicMock()
            mock_cursor.to_list = AsyncMock(
                return_value=[
                    {
                        "key": "key1",
                        "value": "value1",
                        "dataset_id": "dataset1",
                    },
                    {
                        "key": "key2",
                        "value": "value2",
                        "dataset_id": "dataset2",
                    },
                    {
                        "key": "key3",
                        "value": "value3",
                        "dataset_id": "dataset3",
                    },
                ]
            )
            collection.find.return_value = mock_cursor

            # Save the poll time to ensure it's the same one used in the query
            last_poll_time = service._last_poll_time

            # Add a small delay to ensure time passes between poll calls
            await asyncio.sleep(0.001)

            # Call the poll method
            await service._poll()

            # Verify distinct was called
            collection.distinct.assert_called_with(
                "key", {"store_name": "test_store"}
            )

            # Don't directly compare the query since timestamps may have microsecond differences
            assert collection.find.call_count == 1
            call_args = collection.find.call_args[0][0]
            assert call_args["store_name"] == "test_store"
            assert "$gt" in call_args["updated_at"]

            # Verify _last_keys was updated
            assert service._last_keys["test_store"] == current_keys

            # Verify _last_poll_time was updated
            assert service._last_poll_time is not None
            assert service._last_poll_time != last_poll_time

    @pytest.mark.asyncio
    async def test_poll_skips_a_document_that_fails_to_notify(
        self, notification_service
    ):
        service, collection, remote_notifier = notification_service
        docs = _mock_store_collection(collection)

        # The first poll only records the high-water mark and the keys
        await service._poll()

        await asyncio.sleep(0.002)
        updated_at = datetime.datetime.now(datetime.timezone.utc)
        docs.extend(
            [
                _store_doc("bad", {"x": 1}, updated_at),
                _store_doc("good", "ok", updated_at),
            ]
        )
        await asyncio.sleep(0.002)

        notified = []

        async def notify(store_name, message_data):
            notified.append(message_data.key)
            if message_data.key == "bad":
                raise TypeError("Object is not JSON serializable")

        logger_path = "fiftyone.operators.store.notification_service.logger"
        with patch.object(service, "notify", side_effect=notify), patch(
            logger_path
        ) as logger:
            await service._poll()
            last_poll_time = service._last_poll_time
            await service._poll()

        # The document after the failing one is still delivered, the high-
        # water mark moves past both, and the next poll re-notifies neither
        assert notified == ["bad", "good"]
        assert last_poll_time > updated_at
        assert service._last_poll_time > last_poll_time

        # One short warning that names the document, without a traceback
        logger.exception.assert_not_called()
        assert logger.warning.call_count == 1
        args, kwargs = logger.warning.call_args
        message = args[0] % args[1:]
        assert "test_store" in message and "bad" in message
        assert "Object is not JSON serializable" in message
        assert "exc_info" not in kwargs

    @pytest.mark.asyncio
    async def test_poll_notifies_bson_values(self, notification_service):
        service, collection, remote_notifier = notification_service
        docs = _mock_store_collection(collection)
        callback = MagicMock()
        service.subscribe("test_store", callback)

        await service._poll()

        await asyncio.sleep(0.002)
        updated_at = datetime.datetime.now(datetime.timezone.utc)
        oid = ObjectId()
        docs.extend(
            [
                _store_doc(
                    "dated",
                    {"at": datetime.datetime(2026, 1, 1), "id": oid},
                    updated_at,
                ),
                # BSON types that JSON cannot encode still fail, but only
                # their own notification
                _store_doc("decimal", Decimal128("1.5"), updated_at),
                _store_doc("plain", 1, updated_at),
            ]
        )
        await asyncio.sleep(0.002)

        await service._poll()
        await asyncio.gather(*service._background_tasks)
        await service._poll()
        await asyncio.gather(*service._background_tasks)

        # Local subscribers receive every document once
        keys = [c.args[0].key for c in callback.call_args_list]
        assert keys == ["dated", "decimal", "plain"]

        # Remote listeners receive the values that JSON can encode, once
        payloads = {}
        for c in remote_notifier.broadcast_to_store.call_args_list:
            store_name, payload = c.args
            assert store_name == "test_store"
            data = json.loads(payload)
            assert data["key"] not in payloads
            payloads[data["key"]] = data["value"]

        assert sorted(payloads) == ["dated", "plain"]
        assert payloads["dated"] == {
            "at": {"_cls": "DateTime", "datetime": 1767225600000},
            "id": str(oid),
        }
        assert payloads["plain"] == 1

    @pytest.mark.asyncio
    async def test_run_falls_back_to_polling_outside_the_except_block(
        self, notification_service
    ):
        service, collection, remote_notifier = notification_service

        with patch.object(
            service,
            "_run_change_stream",
            side_effect=OperationFailure("Change stream not available"),
        ), patch.object(
            service, "_start_polling", side_effect=ValueError("poll failed")
        ):
            with pytest.raises(ValueError) as e:
                await service._run()

        # Errors raised while polling must not chain the change stream's
        # failure, which would double the tracebacks that polling logs
        assert e.value.__context__ is None


class TestMessageData(unittest.TestCase):
    def test_to_json_encodes_bson_values(self):
        oid = ObjectId()
        dt = datetime.datetime(2026, 1, 1, 12, 30)
        value = {"at": dt, "ids": [oid], "nested": {"at": dt}}
        message = MessageData(
            key="key", value=value, metadata=MessageMetadata()
        )

        data = json.loads(message.to_json())

        dt_json = {"_cls": "DateTime", "datetime": 1767270600000}
        self.assertEqual(
            data["value"],
            {"at": dt_json, "ids": [str(oid)], "nested": {"at": dt_json}},
        )

        # The message's own value is left as is
        self.assertIs(message.value["at"], dt)
        self.assertIs(message.value["ids"][0], oid)

    def test_to_json_encodes_bson_scalar_values(self):
        oid = ObjectId()
        for value, expected in (
            (oid, str(oid)),
            (
                datetime.datetime(2026, 1, 1),
                {"_cls": "DateTime", "datetime": 1767225600000},
            ),
        ):
            message = MessageData(
                key="key", value=value, metadata=MessageMetadata()
            )
            self.assertEqual(json.loads(message.to_json())["value"], expected)


def _store_doc(key, value, updated_at, store_name="test_store"):
    return {
        "store_name": store_name,
        "key": key,
        "value": value,
        "updated_at": updated_at,
    }


def _mock_store_collection(collection):
    """Makes a mock execution store collection answer the poller's queries
    from a list of documents, which the caller fills.
    """
    docs = []

    def _matches(doc, query):
        for field, condition in (query or {}).items():
            if isinstance(condition, dict):
                if not doc[field] > condition["$gt"]:
                    return False
            elif doc[field] != condition:
                return False

        return True

    async def distinct(field, query=None):
        return sorted({d[field] for d in docs if _matches(d, query)})

    def find(query):
        cursor = MagicMock()
        cursor.to_list = AsyncMock(
            return_value=[d for d in docs if _matches(d, query)]
        )
        return cursor

    collection.distinct = AsyncMock(side_effect=distinct)
    collection.find = MagicMock(side_effect=find)
    return docs
