"""Tests for S3 client utilities (mocked boto3)."""

from unittest.mock import MagicMock, patch

from src.utils.s3_client import ensure_bucket_exists, get_s3_client, list_objects, upload_bytes


@patch("src.utils.s3_client.boto3.session.Session")
def test_get_s3_client_returns_client(mock_session_cls):
    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session
    client = get_s3_client()
    mock_session.client.assert_called_once()
    assert client is mock_session.client.return_value


@patch("src.utils.s3_client.get_s3_client")
def test_ensure_bucket_exists_creates_when_missing(mock_get):
    mock_client = MagicMock()
    mock_client.list_buckets.return_value = {"Buckets": []}
    mock_get.return_value = mock_client

    ensure_bucket_exists("test-bucket")
    mock_client.create_bucket.assert_called_once_with(Bucket="test-bucket")


@patch("src.utils.s3_client.get_s3_client")
def test_ensure_bucket_exists_skips_when_present(mock_get):
    mock_client = MagicMock()
    mock_client.list_buckets.return_value = {"Buckets": [{"Name": "test-bucket"}]}
    mock_get.return_value = mock_client

    ensure_bucket_exists("test-bucket")
    mock_client.create_bucket.assert_not_called()


@patch("src.utils.s3_client.get_s3_client")
def test_upload_bytes_puts_object(mock_get):
    mock_client = MagicMock()
    mock_get.return_value = mock_client

    upload_bytes("some/key.jsonl", b"hello", bucket="b")
    mock_client.put_object.assert_called_once_with(Bucket="b", Key="some/key.jsonl", Body=b"hello")


@patch("src.utils.s3_client.get_s3_client")
def test_list_objects_returns_contents(mock_get):
    mock_client = MagicMock()
    mock_client.list_objects_v2.return_value = {"Contents": [{"Key": "a"}, {"Key": "b"}]}
    mock_get.return_value = mock_client

    result = list_objects("prefix/", bucket="b")
    assert len(result) == 2


@patch("src.utils.s3_client.get_s3_client")
def test_list_objects_empty(mock_get):
    mock_client = MagicMock()
    mock_client.list_objects_v2.return_value = {}
    mock_get.return_value = mock_client

    result = list_objects("prefix/", bucket="b")
    assert result == []
